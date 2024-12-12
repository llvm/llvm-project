FROM docker.io/library/ubuntu:22.04 as base
ENV LLVM_SYSROOT=/opt/llvm

FROM base as stage1-toolchain
ENV LLVM_VERSION=19.1.2

RUN apt-get update && \
    apt-get install -y \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    python3 \
    git \
    curl

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

FROM base
    
COPY --from=stage1-toolchain $LLVM_SYSROOT $LLVM_SYSROOT
    
# Need to install curl for hendrikmuhs/ccache-action
# Need nodejs for some of the GitHub actions.
# Need perl-modules for clang analyzer tests.
# Need git for SPIRV-Tools tests.
RUN apt-get update && \
    apt-get install -y \
    binutils \ 
    cmake \
    curl \
    git \
    libstdc++-11-dev \
    ninja-build \
    nodejs \
    perl-modules \
    python3-psutil

ENV LLVM_SYSROOT=$LLVM_SYSROOT
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}
