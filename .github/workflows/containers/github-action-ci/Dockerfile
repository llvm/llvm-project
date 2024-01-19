FROM docker.io/library/ubuntu:22.04 as base
ENV LLVM_SYSROOT=/opt/llvm/

FROM base as toolchain
ENV LLVM_MAJOR=17
ENV LLVM_VERSION=${LLVM_MAJOR}.0.6
ENV LLVM_DIRNAME=clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-22.04
ENV LLVM_FILENAME=${LLVM_DIRNAME}.tar.xz

RUN apt-get update && \
    apt-get install -y \
    curl \
    xz-utils

RUN mkdir -p $LLVM_SYSROOT/bin/ $LLVM_SYSROOT/lib/

RUN curl -O -L https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/$LLVM_FILENAME

RUN tar -C $LLVM_SYSROOT --strip-components=1 -xJf $LLVM_FILENAME \
    $LLVM_DIRNAME/bin/clang \
    $LLVM_DIRNAME/bin/clang++ \
    $LLVM_DIRNAME/bin/clang-cl \
    $LLVM_DIRNAME/bin/clang-$LLVM_MAJOR \
    $LLVM_DIRNAME/bin/lld \
    $LLVM_DIRNAME/bin/ld.lld \
    $LLVM_DIRNAME/lib/clang/


FROM base

COPY --from=toolchain $LLVM_SYSROOT $LLVM_SYSROOT

# Need to install curl for hendrikmuhs/ccache-action
# Need nodejs for some of the GitHub actions.
# Need perl-modules for clang analyzer tests.
RUN apt-get update && \
    apt-get install -y \
    binutils \
    cmake \
    curl \
    libstdc++-11-dev \
    ninja-build \
    nodejs \
    perl-modules \
    python3-psutil

ENV LLVM_SYSROOT=$LLVM_SYSROOT
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}
