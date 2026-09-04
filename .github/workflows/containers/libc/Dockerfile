FROM docker.io/library/ubuntu:24.04

# MPFR is required by some of the mathlib tests.
RUN apt-get update && \
    apt-get install -y \
    git \
    libmpfr-dev \
    libgmp-dev \
    libmpc-dev \
    ninja-build \
    build-essential \
    gcc-9 \
    g++-9 \
    gcc-11 \
    g++-11 \
    qemu-user-static \
    libc6-dev-arm64-cross \
    libstdc++6-arm64-cross \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-riscv64-cross \
    libstdc++6-riscv64-cross \
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    libc6-dev-ppc64el-cross \
    libstdc++6-ppc64el-cross \
    gcc-powerpc64le-linux-gnu \
    g++-powerpc64le-linux-gnu \
    gcc-12-arm-linux-gnueabihf \
    g++-12-arm-linux-gnueabihf \
    sudo \
    sccache \
    zstd \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    cmake \
    m4 \
    linux-libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The downloaded Hexagon QEMU binaries are x86_64-only. Install their runtime
# dependencies only on x86_64 hosts.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    apt-get update && \
    apt-get install -y \
    libpixman-1-0 \
    libslirp0 \
    libfdt1 \
    libglib2.0-0 \
    libfuse3-3 \
    libiscsi7 \
    libaio1t64 \
    liburing2 \
    libcurl4 \
    libcurl3t64-gnutls \
    libasound2t64 \
    libpulse0 \
    libjack-jackd2-0 \
    libsndio7.0 \
    libsndfile1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libsystemd0 \
    libapparmor1 && \
    ln -s "$(dpkg -L libaio1t64 | grep '/libaio.so.1t64$')" \
    "$(dirname "$(dpkg -L libaio1t64 | grep '/libaio.so.1t64$')")/libaio.so.1" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# TODO: Switch to distribution QEMU packages once the Qualcomm fork patches
# are upstreamed to QEMU.
# The current QEMU binaries are x86_64-only, so do not install them on ARM.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    wget --remote-encoding=utf-8 -O /tmp/clang+llvm-23.1.0-rc1-cross-hexagon-unknown-linux-musl.tar.zst \
    https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/23.1.0-rc1/clang+llvm-23.1.0-rc1-cross-hexagon-unknown-linux-musl.tar.zst && \
    mkdir -p /tmp/hexagon-toolchain && \
    tar -xf /tmp/clang+llvm-23.1.0-rc1-cross-hexagon-unknown-linux-musl.tar.zst -C /tmp/hexagon-toolchain/ && \
    install "$(find /tmp/hexagon-toolchain -type f -name qemu-system-hexagon -print -quit)" \
    /usr/local/bin/qemu-system-hexagon && \
    install "$(find /tmp/hexagon-toolchain -type f -name qemu-hexagon -print -quit)" \
    /usr/local/bin/qemu-hexagon && \
    qemu-system-hexagon --version && \
    qemu-hexagon --version && \
    rm -rf /tmp/hexagon-toolchain && \
    rm -f /tmp/clang+llvm-23.1.0-rc1-cross-hexagon-unknown-linux-musl.tar.zst; \
    fi

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 23 && \
    rm llvm.sh

# Install the LLVM testing tools, which the llvm.sh script above does not
# include. This provides lit along with FileCheck, not, count and split-file,
# which are required to run the libc test suites and some CIs like precommit
# testing CI for libc-based compiler-rt builtins. The package only symlinks
# the tools into /usr/bin under versioned names (lit-23, FileCheck-23, ...),
# so also symlink them under their canonical names.
# llvm-link is required to build bitcode library for GPU targets
RUN apt-get install -y llvm-23-tools && \
    ln -s -t /usr/bin $(dpkg -L llvm-23-tools | grep '^/usr/lib/llvm-23/bin/') && \
    ln -s /usr/lib/llvm-23/bin/llvm-link /usr/bin/llvm-link

# Install gmp and mpfr for cross compilation
RUN wget https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz && \
    tar -xf gmp-6.3.0.tar.xz && \
    cd gmp-6.3.0 && \
    ./configure --host=riscv64-linux-gnu --prefix=/usr/riscv64-linux-gnu && \
    make -j8 && \
    sudo make install && \
    cd .. && \
    rm -rf gmp-6.3.0 && \
    rm gmp-6.3.0.tar.xz

RUN wget https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.2.tar.xz && \
    tar -xf mpfr-4.2.2.tar.xz && \
    cd mpfr-4.2.2 && \
    ./configure --host=riscv64-linux-gnu --prefix=/usr/riscv64-linux-gnu \
    --with-gmp=/usr/riscv64-linux-gnu && \
    make -j8 && \
    sudo make install && \
    cd .. && \
    rm -rf mpfr-4.2.2 && \
    rm mpfr-4.2.2.tar.xz

RUN wget https://ftp.gnu.org/gnu/mpc/mpc-1.4.1.tar.xz && \
    tar -xf mpc-1.4.1.tar.xz && \
    cd mpc-1.4.1 && \
    ./configure --host=riscv64-linux-gnu --prefix=/usr/riscv64-linux-gnu \
    --with-gmp=/usr/riscv64-linux-gnu \
    --with-mpfr=/usr/riscv64-linux-gnu && \
    make -j8 && \
    sudo make install && \
    cd .. && \
    rm -rf mpc-1.4.1 && \
    rm mpc-1.4.1.tar.xz

# Install gcc-7 and g++-7.
RUN wget https://ftp.gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.xz && \
    tar -xf gcc-7.5.0.tar.xz && \
    cd gcc-7.5.0 && \
    ./contrib/download_prerequisites && \
    cd .. && \
    mkdir gcc-7 && \
    cd gcc-7 && \
    ../gcc-7.5.0/configure --prefix=/usr/local/gcc7 --enable-languages=c,c++ \
    --disable-multilib --disable-libsanitizer && \
    make -j8 && \
    sudo make install && \
    cd .. && \
    rm -rf gcc-7 && \
    rm -rf gcc-7.5.0 && \
    rm gcc-7.5.0.tar.xz

# Install gcc-8 and g++-8.
RUN wget https://ftp.gnu.org/gnu/gcc/gcc-8.5.0/gcc-8.5.0.tar.xz && \
    tar -xf gcc-8.5.0.tar.xz && \
    cd gcc-8.5.0 && \
    ./contrib/download_prerequisites && \
    cd .. && \
    mkdir gcc-8 && \
    cd gcc-8 && \
    ../gcc-8.5.0/configure --prefix=/usr/local/gcc8 --enable-languages=c,c++ \
    --disable-multilib --disable-libsanitizer && \
    make -j8 && \
    sudo make install && \
    cd .. && \
    rm -rf gcc-8 && \
    rm -rf gcc-8.5.0 && \
    rm gcc-8.5.0.tar.xz

    
# Debian has a multilib setup, so we need to symlink the asm directory.
# For more information, see https://wiki.debian.org/Multiarch/LibraryPathOverview
RUN ln -sf /usr/include/$(uname -p)-linux-gnu/asm /usr/include/asm

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
