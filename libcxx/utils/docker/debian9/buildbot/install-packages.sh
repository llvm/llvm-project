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
      cmake \
      ninja-build \
      curl \
      git \
      gcc-multilib \
      g++-multilib \
      libc6-dev \
      libtool \
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
