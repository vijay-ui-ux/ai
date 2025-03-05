from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from geopy.geocoders import Nominatim
import torch
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import datetime  # Import the datetime module
import requests
# Load SBERT model (lightweight & efficient)
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from PIL import Image
import pytesseract
import io
import os
import re
import vertexai
from vertexai.preview.language_models import TextGenerationModel
import easyocr
import cv2
from google.cloud import vision, translate_v2 as translate
from google.oauth2 import service_account
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from deepmultilingualpunctuation import PunctuationModel
device = torch.device("cpu")

model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)   # Move the model to the device
punctuation_model = PunctuationModel()
summarizer_model = pipeline("summarization", model="Falconsai/text_summarization")
app = FastAPI()
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
# Initialize geopy geolocator
geolocator = Nominatim(user_agent="geoapi")
# Allow frontend access
origins = ["http://localhost:4200"]  # Replace with your Angular app's URL
# origins = ["http://localhost:4200", "https://tech-bot-chi.vercel.app"]  # Replace with your Angular app's URL
# 
# Initialize Vertex AI
vertexai.init(project="cohesive-feat-450806-i9", location="us-central1")  # Replace with your project ID and location


app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Load chat history
chat_history = []

# Load predefined questions & answers
with open("javascript_functions.json", "r") as f:
    javascript_data = json.load(f)

# Prepare stored questions
stored_questions = []
stored_responses = {}
for dataset in ["javascript_functions", "css_data", "family_data", "react_unit_test_cases", "greetings"]:
    stored_questions.extend([item["query"] for item in javascript_data[dataset]])
    stored_responses.update({item["query"]: item for item in javascript_data[dataset]})
# Encode stored questions into vectors
question_embeddings = model.encode(stored_questions, convert_to_tensor=False).astype(np.float32)
# Create FAISS index for fast similarity search
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)
nlp = spacy.load("en_core_web_sm")
# Keyword-based responses
def simple_responses(user_input):
    user_input = user_input.lower()
    doc = nlp(user_input)
    detected_locations = [ent.text.lower() for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    words = user_input.lower().split()
    # print(f"Words: {words}")
    matched_locations = []
    valid_locations = []
    for loc in words:
        try:
            location = geolocator.geocode(loc)
            # print(f"Location: {location.address}")
            if location and "India" in location.address and len(loc) > 2:
                matched_locations.append(loc.title())  # Capitalize for proper format
                # print(f"Matched location: {geolocator.geocode(loc).address}")
        except:
            pass  # Ignore errors

    # print(f"Detected locations: {detected_locations}")
    # print(f"Matched locations: {matched_locations}")
    # valid_locations = set(detected_locations + matched_locations)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]  # GPE is for countries, cities, states
    # print(f"Locations: {locations}")
    keywords = {
        "weather": ["weather", "forecast", "temperature"],
        "farewells": ["bye", "goodbye", "see you later", "farewell", "take care"],
        "thanks": ["thank you", "thanks", "thank you very much", "thank you so much"],
        "name": ["what is your name", "what's your name", "your name"],
        "how_are_you": ["how are you", "how's it going", "are you well"],
        "greetings": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
        "date": ["date", "today's date", "what's the date"],
        "time": ["time", "current time", "what's the time"],
        "help": ["help", "help me", "what can you do", "what can I ask you"]
    }
    responses = {
        "weather": "I'm not a weather expert, but you can check a weather app or website.",
        "farewells": "Goodbye! Have a great day.",
        "thanks": "You're welcome! Feel free to ask me anything.",
        "name": "I am a Tech Bot built to help you with JavaScript, CSS, etc!",
        "how_are_you": "I'm doing great, thank you! How about you?",
        "greetings": "Hi there! How can I help you?",
        "date": datetime.date.today().strftime("%d-%h-%Y"),
        "time": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "help": "I can help you with weather, farewells, thanks, name, how are you, greetings, date, time, and help."
    }

    for key, keyword_list in keywords.items():  # Iterate through key and list of keywords
        if any(word in user_input for word in keyword_list):
            if key == "weather":  # Special handling for weather
                api_key = "9de07940b6b1e197ae7b3614f747308e"  # Replace with your actual API key
                city = matched_locations[0] if matched_locations else "Bengaluru"  # Change or make this dynamic
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    temperature = data["main"]["temp"]
                    description = data["weather"][0]["description"]
                    return f"Current weather in {city} is {temperature}°C with {description}."
                except requests.exceptions.RequestException as e:
                    # print(f"Error fetching weather data: {e}")
                    return "I'm sorry, I couldn't get the weather information right now."
            elif any(word in user_input for word in keyword_list):  # For other keywords
                return responses[key]

    return None

# NLP-based response using SBERT & FAISS
def chatbot_response(user_input):
    # Check for predefined simple responses first
    
    # simple_reply = simple_responses(user_input)
    # if simple_reply:
    #     return {"response": simple_reply, "code_snippet": None}

    # Convert user input into an embedding
    user_embedding = model.encode([user_input]).astype(np.float32)

    # Search for the closest match in FAISS
    D, I = index.search(user_embedding, 1)
    best_match_idx = I[0][0]
    best_similarity = D[0][0]

    # Set a threshold to ensure good matching
    if best_similarity < 10:  # Lower values indicate a closer match in FAISS L2 distance
        best_match_query = stored_questions[best_match_idx]
        best_match_data = stored_responses[best_match_query]
        if best_match_data["response"] == "":
            simple_reply = simple_responses(user_input)
            if simple_reply:
                return {"response": simple_reply, "code_snippet": None}
        return {"response": best_match_data["response"], "code_snippet": best_match_data["code_snippet"]}

    return {"response": "I couldn't find relevant information.", "code_snippet": None}

# image processing response
def image_responses(user_input):
    return {"response": user_input, "code_snippet": None}

# Convert image to NumPy array
def convert_image(image_data):
    """Ensure image is in a format compatible with OpenCV and OCR"""
    img = Image.open(io.BytesIO(image_data))

    # Convert unsupported formats to PNG
    if img.format not in ["JPEG", "PNG"]:
        img = img.convert("RGB")  # Convert to RGB mode
        img = img.save(io.BytesIO(), format="PNG")  # Convert to PNG
        image_data = io.BytesIO().getvalue()

    # Convert PIL image to NumPy array
    img = Image.open(io.BytesIO(image_data))
    return np.array(img)

# Determine if the text is handwritten
def is_handwritten(image):
    """Determine if the text is handwritten by analyzing stroke thickness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Detect edges
    edge_ratio = np.sum(edges) / (edges.shape[0] * edges.shape[1])  # Edge density
    return edge_ratio > 0.08  # Adjust threshold if needed

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        # print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        # print("Could not understand audio")
        return None
    except sr.RequestError as e:
        # print(f"Could not request results from speech recognition service; {e}")
        return None

# Function to detect language using spaCy
def detect_language(text):
    translate_client = translate.Client()
    result = translate_client.detect_language(text)
    # print(f"Detected language: {result['language']}")
    return result['language']
    # doc = nlp_xx(text)  # Use the multilingual model for language detection
    # return doc.lang_

# # Function to translate text using Hugging Face Transformers
# def translate_text(text):
#     translation = translator(text)
#     return translation['translation_text']

# Function to translate text using Google Cloud Translation API
# def translate_text(text, target_language="en"):
#     credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
#     if credentials_json:
#         credentials_dict = json.loads(credentials_json)
#         # Create a client with the credentials
#         translate_client = translate.Client.from_service_account_info(credentials_dict)
#     else:
#         translate_client = translate.Client()
#     result = translate_client.translate(text, target_language=target_language)
#     return result['translatedText']

def translate_text(text, target_language="en"):    
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# def translate_text_to_target(text, target_language):
#     credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
#     if credentials_json:
#         credentials_dict = json.loads(credentials_json)
#         # Create a client with the credentials
#         translate_client = translate.Client.from_service_account_info(credentials_dict)
#     else:
#         translate_client = translate.Client()
#     result = translate_client.translate(text, target_language=target_language)
#     return result['translatedText']

def extract_code_snippet(user_input):
  """Extracts code snippets from user input.

  Args:
    user_input: The user's input string.

  Returns:
    The extracted code snippet as a string, or None if no code snippet is found.
  """

  # Define a regular expression pattern to match code snippets
  # This pattern assumes code snippets are enclosed in triple backticks (```)
  pattern = r"```(?:\w+\n)?(.*?)```"  # Matches code between triple backticks, optionally with a language specifier

  match = re.search(pattern, user_input, re.DOTALL)  # re.DOTALL allows the pattern to match across multiple lines

  if match:
    return match.group(1).strip()  # Extract the code and remove leading/trailing whitespace
  else:
    return None

# Function to analyze and explain code
def analyze_code(code_snippet, language="python"):
    #... (optional preprocessing of the code snippet)
    prompt = f"Analyze and explain the following {language} code:\n\n```{language}\n{code_snippet}\n```"
    response = code_model.predict(prompt)
    # response = code_model.predict(prefix=code_snippet, temperature=0.2, max_output_tokens=256)
    return response.text

# Function to generate text using Gemini
def generate_text(prompt):
    response = genai_model.predict(prompt)
    return response.text


def format_text_to_html(text):
    """
    Replaces **bold** text with <b>bold</b> in a string.

    Args:
        text: The input string.

    Returns:
        The formatted string with <b> tags.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

def generate(prompt, chat_history):
    client = genai.Client(
        vertexai=True,
        project="cohesive-feat-450806-i9",
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"
    contents = chat_history
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_LOW_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_LOW_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_LOW_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_LOW_AND_ABOVE"
            )
        ],
        tools=tools,
    )
    generated_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # if not chunk.candidates or not chunk.candidates[0].content.parts:
        #     continue
        generated_text += chunk.text
    # print(f"gemini response: ", generated_text)
    return generated_text



@app.get("/api/home")
def home():
    return {"message": "FastAPI is running!"}
# CORS preflight request
# @app.options("/{full_path:path}")
# async def preflight(full_path: str):
#     return {"message": "Preflight successful"}


# @app.options("/api/chat")  # Add this to explicitly handle OPTIONS requests
# async def chat_options():
#     response = Response()
#     response.headers["Access-Control-Allow-Origin"] = "https://tech-bot-chi.vercel.app"  # Your frontend origin
#     response.headers["Access-Control-Allow-Methods"] = "POST"  # Allowed methods
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type"  # Allowed headers
#     return response


@app.post("/api/conversation")
async def conversation(request: ChatRequest):
    user_input = request.message[len(request.message) - 1]

# API endpoint for chatbot
@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    # Check if the user is asking for code analysis
    # if "analyze" in user_input.lower() or "explain" in user_input.lower():
    # Extract the code snippet from the user input
    code_snippet = extract_code_snippet(user_input)  # You'll need to implement this function
    # Analyze the code
    # analysis = analyze_code(code_snippet)
    # analysis = generate_text(code_snippet)
    chat_history.append(types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input)
            ]
        ))
    analysis = generate(user_input, chat_history)
    print(f"Analysis result: {analysis}")
    analysis_language = detect_language(analysis)
    title_change = translate_text("TechBot", target_language=analysis_language)
    google_change = translate_text("google", target_language=analysis_language)
    google_change_case = translate_text("Google", target_language=analysis_language)
    analysis = analysis.replace(google_change.lower(), title_change.lower())
    analysis = analysis.replace("Google", "TechBot")
    # print(f"Analysis result: {analysis}")
    if not analysis:
        analysis = "I couldn't analyze the information."
    chat_history.append(types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=analysis)
            ]
        ))
    return {"response": analysis, "code_snippet": None, "chat_history": chat_history}
    # else:
    # print(f"Received message: {request}")
    language = detect_language(user_input)
    # language = detect_language(user_input)
    translation = translate_text(user_input)
    # print(f"Translated text: {translation}")
    # print(f"Detected language: {language}")
    # if language != "en":
    #     user_input_en = translate_text(user_input)
    bot_response = chatbot_response(translation)  # Process user input
    # print(f"Bot response: {bot_response}")
    translated_response = translate_text(bot_response["response"], target_language=language)
    # print(f"Translated response: {translated_response}")
    return {"response": translated_response, "code_snippet": bot_response["code_snippet"]}

@app.post("/api/audioTOText")
async def audio_to_text(video: UploadFile = File(None)):
    # Save the uploaded audio temporarily
    temp_audio_path = "temp_audio.mp3"  # Or use a more general extension like.wav
    with open(temp_audio_path, "wb") as buffer:
        buffer.write(await audio.read())

    # Transcribe audio
    text = transcribe_audio(temp_audio_path)
    if text:
        bot_response = chatbot_response(text)  # Process transcribed text
    else:
        bot_response = {"response": "I couldn't transcribe the audio.", "code_snippet": None}
    return bot_response

# API endpoint for video processing
@app.post("/api/videoTOText")
async def video_to_text(video: UploadFile = File(None)):
    # Save the uploaded video temporarily
    temp_video_path = "temp_video.mp4"
    temp_audio_path = "temp_audio.wav"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await video.read())

    # Extract audio
    extract_audio(temp_video_path, temp_audio_path)

    # Transcribe audio
    text = transcribe_audio(temp_audio_path)
    text = punctuation_model.restore_punctuation(text)
    # text = text['generated_text']  # Extract the punctuated text
    text = text.capitalize()  # Capitalize the first letter
    summarized_text = summarizer_model(text, max_length=1000, min_length=30, do_sample=False)
    if text:
        # print(f"Transcribed text: {summarized_text[0]['summary_text']}")  # Print the transcribed text
        bot_response = {"response": summarized_text[0]['summary_text'], "code_snippet": None}  # Process transcribed text
    else:
        bot_response = {"response": "I couldn't transcribe the audio.", "code_snippet": None}
    return bot_response

# API endpoint for image processing
@app.post("/api/checkImage")
async def checkImage(image: UploadFile = File(None)):
    # print(f"Received image: {image}")
    try:
        contents = await image.read()
        image = convert_image(contents)
        if len(image.shape) == 2:
            image_cv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # ... (image preprocessing if needed)
        if is_handwritten(image_cv):
            text = " ".join(reader.readtext(image, detail=0))  # EasyOCR for handwriting
        else:
            text = pytesseract.image_to_string(image)
        bot_response = image_responses(text)  # Process extracted text
        # print(f"Processed image: {bot_response}")
        return bot_response
    except Exception as e:
        # print(f"Error processing image: {e}")
        bot_response = {"response": "I couldn't process the image. Please try again later.", "code_snippet": None}
        return bot_response

# Run with: uvicorn chatbot:app --reload