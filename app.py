from __future__ import division, print_function
import os

import numpy as np
import tensorflow as tf
import cv2

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/multipleTeamClassifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    ans=yhat.flatten().tolist()
    index=ans.index(max(ans))
    acc=max(ans)
    preds=''
    if index==0:
        preds= 'Arsenal'
    elif index==1:
        preds= 'Chelsea'
    elif index==2:
        preds= 'Liverpool'
    elif index==3:
        preds= 'Man City'
    elif index==4:
        preds= 'Man Utd'
    elif index==5:
        preds= 'Tottenham'
    
    return preds+'----------Accuracy: '+str(acc*100)+'%'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

