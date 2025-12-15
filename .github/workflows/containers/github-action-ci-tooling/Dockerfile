ARG LLVM_VERSION=21.1.0
# FIXME: Use "${LLVM_VERSION%%.*}" instead of "LLVM_VERSION_MAJOR" once we update runners to Ubuntu-26.04 with Buildah >= 1.37
ARG LLVM_VERSION_MAJOR=21

FROM docker.io/library/ubuntu:24.04 AS llvm-downloader
ARG LLVM_VERSION
ARG LLVM_VERSION_MAJOR

RUN apt-get update && \
    apt-get install -y wget xz-utils && \
    wget -O llvm.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/LLVM-${LLVM_VERSION}-Linux-X64.tar.xz && \
    mkdir -p /llvm-extract && \
    tar -xvJf llvm.tar.xz -C /llvm-extract \
        # Only unpack these tools to save space on Github runner.
        LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-${LLVM_VERSION_MAJOR} \
        LLVM-${LLVM_VERSION}-Linux-X64/lib/clang/${LLVM_VERSION_MAJOR}/include \
        LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-tidy \
        LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-format \
        LLVM-${LLVM_VERSION}-Linux-X64/bin/git-clang-format && \
    rm llvm.tar.xz


FROM docker.io/library/ubuntu:24.04 AS base
ENV LLVM_SYSROOT=/opt/llvm
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

# Need nodejs for some of the GitHub actions.
# Need git for git-clang-format.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    nodejs \
    sudo \
    # These are needed by the premerge pipeline.
    # Pip is used to install dependent python packages.
    python3-pip \
    python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a new user with id 1001 as that is the user id that
# Github Actions uses to perform the checkout action.
RUN useradd gha -u 1001 -m -s /bin/bash
RUN adduser gha sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# Don't set USER gha right away because we still need to install packages
# as root in 'ci-container-code-format' and 'ci-container-code-lint' containers


FROM base AS ci-container-build-tools
ARG LLVM_VERSION
ARG LLVM_VERSION_MAJOR

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-${LLVM_VERSION_MAJOR} \
                            ${LLVM_SYSROOT}/bin/
COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/lib/clang/${LLVM_VERSION_MAJOR}/include \
                            ${LLVM_SYSROOT}/lib/clang/${LLVM_VERSION_MAJOR}/include
RUN ln -s ${LLVM_SYSROOT}/bin/clang-${LLVM_VERSION_MAJOR} ${LLVM_SYSROOT}/bin/clang && \
    ln -s ${LLVM_SYSROOT}/bin/clang ${LLVM_SYSROOT}/bin/clang++

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CC=${LLVM_SYSROOT}/bin/clang
ENV CXX=${LLVM_SYSROOT}/bin/clang++


FROM base AS ci-container-code-format
ARG LLVM_VERSION

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-format \
                            /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/git-clang-format \
                            ${LLVM_SYSROOT}/bin/


# Install dependencies for 'pr-code-format.yml' job
COPY llvm/utils/git/requirements_formatting.txt requirements_formatting.txt
RUN pip install -r requirements_formatting.txt --break-system-packages && \
    rm requirements_formatting.txt
USER gha
WORKDIR /home/gha


FROM ci-container-build-tools AS ci-container-code-lint
ARG LLVM_VERSION
ARG LLVM_VERSION_MAJOR

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-tidy \
                            ${LLVM_SYSROOT}/bin/
COPY clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py ${LLVM_SYSROOT}/bin/clang-tidy-diff.py

# Install dependencies for 'pr-code-lint.yml' job
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-doc8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY llvm/utils/git/requirements_linting.txt requirements_linting.txt
RUN pip install -r requirements_linting.txt --break-system-packages && \
    rm requirements_linting.txt
USER gha
WORKDIR /home/gha


FROM ci-container-build-tools as ci-container-abi-tests

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    abi-compliance-checker \
    abi-dumper \
    autoconf \
    parallel \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/universal-ctags/ctags.git && \
    cd ctags && \
    ./autogen.sh && \
    ./configure && \
    sudo make install && \
    rm -Rf ../ctags

