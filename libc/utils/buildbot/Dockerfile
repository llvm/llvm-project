FROM debian:10

# Installing dependencies.
RUN dpkg --add-architecture i386
RUN apt-get update
RUN apt-get install -y build-essential clang subversion git vim \
  zip libstdc++6:i386 file binutils-dev binutils-gold cmake python-pip \
  ninja-build
RUN python -m pip install buildbot-worker==2.8.4

# Temporary dependencies for AOR tests.
RUN apt-get install -y libmpfr-dev libmpc-dev

# Change linker to gold.
RUN update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.gold" 20

# Create and switch to buildbot user.
RUN useradd buildbot --create-home
USER buildbot

WORKDIR /home/buildbot

# Use clang as the compiler.
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

ENV WORKER_NAME="libc-x86_64-debian"

# Set up buildbot host and maintainer info.
RUN mkdir -p "${WORKER_NAME}/info/"
RUN bash -c "(uname -a ; \
  gcc --version | head -n1 ; ld --version \
  | head -n1 ; cmake --version | head -n1 ) > ${WORKER_NAME}/info/host"
RUN echo "Paula Toth <paulatoth@google.com>" > "${WORKER_NAME}/info/admin"

ADD --chown=buildbot:buildbot run.sh .
ENTRYPOINT ["./run.sh"]
