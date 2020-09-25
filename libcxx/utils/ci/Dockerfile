#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This Dockerfile describes the base image used to run the various libc++
# build bots. By default, the image runs the Buildkite Agent, however one
# can also just start the image with a shell to debug CI failures.
#
# To start a Buildkite Agent, run it as:
#   $ docker run --env-file secrets.env -it $(docker build -q .)
#
# The environment variables in `secrets.env` must be replaced by the actual
# tokens for this to work. These should obviously never be checked in.
#
# To start a shell in the Docker image, use:
#   $ docker run -it --volume "$(git rev-parse --show-toplevel):/llvm" --workdir "/llvm" $(docker build -q .) bash
#
# This will fire up the Docker container and mount the root of the monorepo
# as /llvm in the container. Be careful, the state in /llvm is shared between
# the container and the host machine.
#

FROM ubuntu:bionic

RUN apt-get update
RUN apt-get install -y bash curl

# Install the most recently released LLVM
RUN apt-get install -y lsb-release wget software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
RUN ln -s $(find /usr/bin -regex '.+/clang\+\+-[a-zA-Z0-9.]+') /usr/bin/clang++
RUN ln -s $(find /usr/bin -regex '.+/clang-[a-zA-Z0-9.]+') /usr/bin/clang

# Install a recent GCC
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt install -y gcc-10 g++-10
RUN ln -s $(find /usr/bin -regex '.+/g\+\+-[a-zA-Z0-9.]+') /usr/bin/g++
RUN ln -s $(find /usr/bin -regex '.+/gcc-[a-zA-Z0-9.]+') /usr/bin/gcc

# Install a recent CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh -O /tmp/install-cmake.sh
RUN bash /tmp/install-cmake.sh --prefix=/usr --exclude-subdir --skip-license
RUN rm /tmp/install-cmake.sh

# Install other tools used by the build or the test suite
RUN apt-get install -y ninja-build python3 sphinx-doc

# Install the Buildkite agent and dependencies
RUN bash -c "$(curl -sL https://raw.githubusercontent.com/buildkite/agent/master/install.sh)"
RUN apt-get install -y git
ENV PATH="${PATH}:/root/.buildkite-agent/bin"

# Install the Phabricator Python module to allow uploading results to Phabricator
RUN apt-get install -y python3-pip
RUN pip3 install phabricator

# By default, start the Buildkite agent (this requires a token).
CMD buildkite-agent start --tags "queue=libcxx-builders"
