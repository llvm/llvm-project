FROM docker.io/library/ubuntu:22.04 as base
ENV LLVM_SYSROOT=/opt/llvm

FROM base as stage1-toolchain
ENV LLVM_VERSION=19.1.5

RUN apt-get update && \
    apt-get install -y \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    python3 \
    git \
    curl \
    zlib1g-dev

RUN curl -O -L https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-$LLVM_VERSION.tar.gz && tar -xf llvmorg-$LLVM_VERSION.tar.gz

WORKDIR /llvm-project-llvmorg-$LLVM_VERSION

# Patch to enable better PGO profile data.
# TODO: Remove this for llvm 20
ADD https://github.com/llvm/llvm-project/commit/738250989ce516f02f809bdfde474a039c77e81f.patch .

RUN patch -p1 < 738250989ce516f02f809bdfde474a039c77e81f.patch

RUN cmake -B ./build -G Ninja ./llvm \
  -C ./clang/cmake/caches/BOLT-PGO.cmake \
  -DBOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DBOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DPGO_INSTRUMENT_LTO=Thin \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_INSTALL_PREFIX="$LLVM_SYSROOT" \
  -DLLVM_ENABLE_PROJECTS="bolt;clang;lld;clang-tools-extra" \
  -DLLVM_DISTRIBUTION_COMPONENTS="lld;compiler-rt;clang-format;scan-build" \
  -DCLANG_DEFAULT_LINKER="lld"

RUN ninja -C ./build stage2-clang-bolt stage2-install-distribution && ninja -C ./build install-distribution

FROM base as ci-container
    
COPY --from=stage1-toolchain $LLVM_SYSROOT $LLVM_SYSROOT

# Need to install curl for hendrikmuhs/ccache-action
# Need nodejs for some of the GitHub actions.
# Need perl-modules for clang analyzer tests.
# Need git for SPIRV-Tools tests.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    binutils \ 
    cmake \
    curl \
    git \
    libstdc++-11-dev \
    ninja-build \
    nodejs \
    perl-modules \
    python3-psutil \
    sudo \

    # These are needed by the premerge pipeline. Pip is used to install
    # dependent python packages and ccache is used for build caching. File and
    # tzdata are used for tests.
    python3-pip \
    ccache \
    file \
    tzdata

# Install sccache as it is needed by most of the project test workflows and
# cannot be installed by the ccache action when executing as a non-root user.
# TODO(boomanaiden154): This should be switched to being installed with apt
# once we bump to Ubuntu 24.04.
RUN curl -L 'https://github.com/mozilla/sccache/releases/download/v0.7.6/sccache-v0.7.6-x86_64-unknown-linux-musl.tar.gz' > /tmp/sccache.tar.gz && \
    echo "2902a5e44c3342132f07b62e70cca75d9b23252922faf3b924f449808cc1ae58 /tmp/sccache.tar.gz" | sha256sum -c && \
    tar xzf /tmp/sccache.tar.gz -O --wildcards '*/sccache' > '/usr/local/bin/sccache' && \
    rm /tmp/sccache.tar.gz && \
    chmod +x /usr/local/bin/sccache

ENV LLVM_SYSROOT=$LLVM_SYSROOT
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

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

FROM ci-container as ci-container-agent

ENV GITHUB_RUNNER_VERSION=2.322.0

RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -O -L https://github.com/actions/runner/releases/download/v$GITHUB_RUNNER_VERSION/actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    tar xzf ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    rm ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz

