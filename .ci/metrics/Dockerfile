FROM docker.io/python:3.12

COPY requirements.lock.txt ./
RUN pip3 install --no-cache-dir -r requirements.lock.txt
COPY metrics.py ./

CMD ["python3", "metrics.py"]
