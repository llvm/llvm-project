FROM docker.io/python:3.15@sha256:2e13bb9164213e8dec5adf78bc63bb710054c4b656800e29a25f756ed2fa18ae

COPY requirements.lock.txt ./
RUN pip3 install --require-hashes --no-cache-dir -r requirements.lock.txt
COPY metrics.py ./

CMD ["python3", "metrics.py"]
