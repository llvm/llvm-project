This folder contains resources needed to create a docker container for
llvm-libc buildbot worker.

Dockerfile: Sets up the docker image with all pre-requisites.

run.sh: Script to create and start buildbot worker with supplied password.

cmd to build the docker container:
```
docker build -t llvm-libc-buildbot-worker .
```

cmd to run the buildbot:
```
docker run -it llvm-libc-buildbot-worker <passwd>
```
