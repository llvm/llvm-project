FROM docker.io/library/ubuntu:24.04 AS llvm-downloader

ENV LLVM_VERSION=21.1.2

RUN apt-get update && \
    apt-get install -y wget pixz && \
    wget --progress=bar:force -O llvm.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/LLVM-${LLVM_VERSION}-Linux-X64.tar.xz && \
    mkdir -p /llvm-extract && \
    pixz -d llvm.tar.xz && \
    tar -xvf llvm.tar -C /llvm-extract LLVM-${LLVM_VERSION}-Linux-X64/bin/ && \
    rm llvm.tar

FROM docker.io/library/ubuntu:24.04 AS base

ENV LLVM_SYSROOT=/opt/llvm

# Need nodejs for some of the GitHub actions.
# Need git for git-clang-format.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    nodejs \
    sudo \
    # These are needed by the premerge pipeline. Pip and venv are used to
    # install dependent python packages.
    # Having a symlink from python to python3 enables code sharing between
    # the Linux and Windows pipelines.
    python3-pip \
    python3-venv \
    python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Create a new user to avoid test failures related to a lack of expected
# permissions issues in some tests. Set the user id to 1001 as that is the
# user id that Github Actions uses to perform the checkout action.
RUN useradd gha -u 1001 -m -s /bin/bash

# Also add the user to passwordless sudoers so that we can install software
# later on without having to rebuild the container.
RUN adduser gha sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER gha
WORKDIR /home/gha

FROM base AS ci-container-code-format

ENV LLVM_VERSION=21.1.2

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-format ${LLVM_SYSROOT}/bin/clang-format

ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

# Install dependencies for 'pr-code-format.yml' job
COPY llvm/utils/git/requirements_formatting.txt /home/gha/requirements_formatting.txt
RUN python -m venv venv && \
    venv/bin/pip install -r /home/gha/requirements_formatting.txt && \
    rm /home/gha/requirements_formatting.txt

# Stage 4: clang-lint container
FROM base AS ci-container-code-lint

ENV LLVM_VERSION=21.1.2

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-tidy ${LLVM_SYSROOT}/bin/
COPY clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py ${LLVM_SYSROOT}/bin/clang-tidy-diff.py

ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

COPY llvm/utils/git/requirements_linting.txt /home/gha/requirements_linting.txt
RUN python -m venv venv && \
    venv/bin/pip install -r /home/gha/requirements_linting.txt && \
    rm /home/gha/requirements_linting.txt
