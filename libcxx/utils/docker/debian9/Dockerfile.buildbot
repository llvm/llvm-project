
#===-------------------------------------------------------------------------------------------===//
# buildslave
#===-------------------------------------------------------------------------------------------===//
FROM ericwf/builder-base:latest AS buildbot

# Copy over the GCC and Clang installations
COPY --from=ericwf/gcc-5:latest /opt/gcc-5 /opt/gcc-5
COPY --from=ericwf/gcc-tot:latest /opt/gcc-tot /opt/gcc-tot
COPY --from=ericwf/llvm-4:latest /opt/llvm-4.0 /opt/llvm-4.0

RUN ln -s /opt/gcc-5/bin/gcc /usr/local/bin/gcc-4.9 && \
    ln -s /opt/gcc-5/bin/g++ /usr/local/bin/g++-4.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash-completion \
    vim \
    sudo \
    buildbot-slave \
  && rm -rf /var/lib/apt/lists/*

ADD scripts/install_clang_packages.sh /tmp/install_clang_packages.sh
RUN /tmp/install_clang_packages.sh && rm /tmp/install_clang_packages.sh

RUN git clone https://git.llvm.org/git/libcxx.git /libcxx
