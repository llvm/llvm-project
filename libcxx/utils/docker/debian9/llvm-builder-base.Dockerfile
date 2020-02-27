#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//

FROM launcher.gcr.io/google/debian9:latest AS llvm-builder-base
LABEL maintainer "libc++ Developers"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      gnupg \
      build-essential \
      wget \
      subversion \
      unzip \
      automake \
      python \
      cmake \
      ninja-build \
      curl \
      git \
      gcc-multilib \
      g++-multilib \
      libc6-dev \
      bison \
      flex \
      libtool \
      autoconf \
      binutils-dev \
      binutils-gold \
      software-properties-common \
      gnupg \
      apt-transport-https \
      sudo \
      bash-completion \
      vim \
      systemd \
      sysvinit-utils \
      systemd-sysv && \
  update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.gold" 20 && \
  update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.bfd" 10 && \
  rm -rf /var/lib/apt/lists/*

