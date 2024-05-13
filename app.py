import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import io
from PyPDF2 import PdfReader  # Import PdfReader from PyPDF2

# Load the trained Neural Network model
nn_model = load_model('nn_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define function to preprocess text
def preprocess_text(text):
    # Tokenize text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=200)
    return padded_sequence

# Define function to read PDF file
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit app
def main():
    st.write("""
Members:
- Gurira Wesley P R204433P HAI
- Sendekera Cousins R207642E HAI
- Ryan Kembo R205723E HAI
- Cyprian Masvikeni R205537V HDSC
""")

    st.title("News Prediction")

    st.write("Upload a PDF or TXT file containing news text to predict its veracity (True or False).")

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            # Read PDF file
            raw_text = read_pdf(uploaded_file)
        elif uploaded_file.type == 'text/plain':
            # Read text file
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")  # Specify encoding and handle errors


        st.header("Original Text")
        st.text(raw_text)

        st.header("Prediction")

        # Preprocess text
        processed_text = preprocess_text(raw_text)

        # Make prediction using the Neural Network model
        prediction = nn_model.predict(processed_text)
        label = "True" if prediction[0][0] > 0.5 else "False"

        st.write(f"The news is predicted to be: {label}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
