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


FROM base AS ci-container-code-format
ARG LLVM_VERSION

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-format \
                            /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/git-clang-format \
                            ${LLVM_SYSROOT}/bin/

ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

# Install dependencies for 'pr-code-format.yml' job
COPY llvm/utils/git/requirements_formatting.txt requirements_formatting.txt
RUN pip install -r requirements_formatting.txt --break-system-packages && \
    rm requirements_formatting.txt


FROM base AS ci-container-code-lint
ARG LLVM_VERSION
ARG LLVM_VERSION_MAJOR

COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-tidy \
                            /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/bin/clang-${LLVM_VERSION_MAJOR} \
                            ${LLVM_SYSROOT}/bin/
COPY --from=llvm-downloader /llvm-extract/LLVM-${LLVM_VERSION}-Linux-X64/lib/clang/${LLVM_VERSION_MAJOR}/include \
                            ${LLVM_SYSROOT}/lib/clang/${LLVM_VERSION_MAJOR}/include
COPY clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py ${LLVM_SYSROOT}/bin/clang-tidy-diff.py

RUN ln -s ${LLVM_SYSROOT}/bin/clang-${LLVM_VERSION_MAJOR} ${LLVM_SYSROOT}/bin/clang && \
    ln -s ${LLVM_SYSROOT}/bin/clang ${LLVM_SYSROOT}/bin/clang++

ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for 'pr-code-lint.yml' job
COPY llvm/utils/git/requirements_linting.txt requirements_linting.txt
RUN pip install -r requirements_linting.txt --break-system-packages && \
    rm requirements_linting.txt
