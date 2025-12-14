# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This file defines the image we use for Linux testing using Github Actions.
# From the root of the monorepo, this image can be built with:
#
#   $ docker build --file libcxx/utils/ci/docker/linux-builder.dockerfile \
#                  --build-arg BASE_IMAGE_VERSION=<sha>                   \
#                  --build-arg GITHUB_RUNNER_VERSION=<version> .
#
# This image also gets built on every push to `main` that modifies these Docker
# files, and can be found at ghcr.io/llvm/libcxx-linux-builder.

ARG BASE_IMAGE_VERSION
FROM ghcr.io/llvm/libcxx-linux-builder-base:${BASE_IMAGE_VERSION}

ARG GITHUB_RUNNER_VERSION

# Setup the user
RUN useradd gha -u 1001 -m -s /bin/bash
RUN adduser gha sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/gha
USER gha

# Install the Github Actions runner
ENV RUNNER_MANUALLY_TRAP_SIG=1
ENV ACTIONS_RUNNER_PRINT_LOG_TO_STDOUT=1
RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -O -L https://github.com/actions/runner/releases/download/v$GITHUB_RUNNER_VERSION/actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    tar xzf ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    rm ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz
