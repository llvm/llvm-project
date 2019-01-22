#===- libcxx/utils/docker/debian9/Dockerfile --------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-------------------------------------------------------------------------------------------===//

# Setup the base builder image with the packages we'll need to build GCC and Clang from source.
FROM launcher.gcr.io/google/debian9:latest AS builder-base
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
      systemd \
      sysvinit-utils && \
  update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.gold" 20 && \
  update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.bfd" 10 && \
  rm -rf /var/lib/apt/lists/*


# Build GCC versions
FROM builder-base as gcc-48-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-4.8.5 --branch gcc-4_8_5-release \
    --cherry-pick ec1cc0263f156f70693a62cf17b254a0029f4852

FROM builder-base as gcc-49-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-4.9.4 --branch gcc-4_9_4-release

FROM builder-base as gcc-5-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-5 --branch gcc-5_5_0-release

FROM builder-base as gcc-6-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-6 --branch gcc-6_5_0-release

FROM builder-base as gcc-7-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-7 --branch gcc-7_4_0-release

FROM builder-base as gcc-8-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-8 --branch gcc-8_2_0-release

FROM builder-base as gcc-tot-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_gcc_version.sh /tmp/build_gcc_version.sh
RUN /tmp/build_gcc_version.sh --install /opt/gcc-tot --branch master

# Build additional LLVM versions
FROM builder-base as llvm-36-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-3.6 --branch release/3.6.x

FROM builder-base as llvm-37-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-3.7 --branch release/3.7.x

FROM builder-base as llvm-38-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-3.8 --branch release/3.8.x

FROM builder-base as llvm-39-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-3.9 --branch release/3.9.x

FROM builder-base as llvm-4-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-4.0 --branch release/4.x

FROM builder-base as llvm-5-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-5.0 --branch release/5.x

FROM builder-base as llvm-6-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-6.0 --branch release/6.x

FROM builder-base as llvm-7-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-7.0 --branch release/7.x

FROM builder-base as llvm-8-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-8.0 --branch release/8.x

FROM builder-base as llvm-tot-builder
LABEL maintainer "libc++ Developers"

ADD scripts/build_llvm_version.sh /tmp/build_llvm_version.sh
RUN /tmp/build_llvm_version.sh --install /opt/llvm-tot --branch master


#===-------------------------------------------------------------------------------------------===//
# buildslave
#===-------------------------------------------------------------------------------------------===//
FROM builder-base AS buildbot

# Copy over the GCC and Clang installations
COPY --from=gcc-49-builder /opt/gcc-4.9.4 /opt/gcc-4.9.4
COPY --from=gcc-tot-builder /opt/gcc-tot /opt/gcc-tot
COPY --from=llvm-4-builder /opt/llvm-4.0 /opt/llvm-4.0

RUN ln -s /opt/gcc-4.9.4/bin/gcc /usr/local/bin/gcc-4.9 && \
    ln -s /opt/gcc-4.9.4/bin/g++ /usr/local/bin/g++-4.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash-completion \
    buildbot-slave \
  && rm -rf /var/lib/apt/lists/*

ADD scripts/install_clang_packages.sh /tmp/install_clang_packages.sh
RUN /tmp/install_clang_packages.sh && rm /tmp/install_clang_packages.sh

RUN git clone https://git.llvm.org/git/libcxx.git /libcxx

#===-------------------------------------------------------------------------------------------===//
#  compiler-zoo
#===-------------------------------------------------------------------------------------------===//
FROM  buildbot AS compiler-zoo
LABEL maintainer "libc++ Developers"

# Copy over the GCC and Clang installations
COPY --from=gcc-48-builder /opt/gcc-4.8.5 /opt/gcc-4.8.5
COPY --from=gcc-49-builder /opt/gcc-4.9.4 /opt/gcc-4.9.4
COPY --from=gcc-5-builder /opt/gcc-5   /opt/gcc-5
COPY --from=gcc-6-builder /opt/gcc-6   /opt/gcc-6
COPY --from=gcc-7-builder /opt/gcc-7   /opt/gcc-7
COPY --from=gcc-8-builder /opt/gcc-8   /opt/gcc-8
COPY --from=gcc-tot-builder /opt/gcc-tot /opt/gcc-tot

COPY --from=llvm-36-builder /opt/llvm-3.6 /opt/llvm-3.6
COPY --from=llvm-37-builder /opt/llvm-3.7 /opt/llvm-3.7
COPY --from=llvm-38-builder /opt/llvm-3.8 /opt/llvm-3.8
COPY --from=llvm-39-builder /opt/llvm-3.9 /opt/llvm-3.9
COPY --from=llvm-4-builder /opt/llvm-4.0 /opt/llvm-4.0
COPY --from=llvm-5-builder /opt/llvm-5.0 /opt/llvm-5.0
COPY --from=llvm-6-builder /opt/llvm-6.0 /opt/llvm-6.0
COPY --from=llvm-7-builder /opt/llvm-7.0 /opt/llvm-7.0
COPY --from=llvm-8-builder /opt/llvm-8.0 /opt/llvm-8.0
COPY --from=llvm-tot-builder /opt/llvm-tot /opt/llvm-tot


