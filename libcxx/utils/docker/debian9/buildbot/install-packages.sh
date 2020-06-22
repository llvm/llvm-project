#!/usr/bin/env bash

set -x
set -e

apt-get update && \
    apt-get install -y --no-install-recommends \
      buildbot-slave \
      ca-certificates \
      gnupg \
      build-essential \
      wget \
      unzip \
      python \
      ninja-build \
      curl \
      git \
      gcc-multilib \
      g++-multilib \
      libc6-dev \
      libtool \
      locales-all \
      binutils-dev \
      binutils-gold \
      software-properties-common \
      gnupg \
      apt-transport-https \
      sudo \
      bash-completion \
      vim \
      jq \
      systemd \
      sysvinit-utils \
      systemd-sysv && \
  rm -rf /var/lib/apt/lists/*

# Install a recent CMake
yes | apt-get purge cmake
wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2-Linux-x86_64.sh -O /tmp/install-cmake.sh
bash /tmp/install-cmake.sh --prefix=/usr --exclude-subdir --skip-license
