
#===-------------------------------------------------------------------------------------------===//
# buildslave
#===-------------------------------------------------------------------------------------------===//
FROM ericwf/llvm-builder-base:latest AS llvm-buildbot-worker

COPY --from=ericwf/compiler:gcc-5 /opt/gcc-5 /opt/gcc-5
COPY --from=ericwf/compiler:gcc-tot /opt/gcc-tot /opt/gcc-tot
COPY --from=ericwf/compiler:llvm-4 /opt/llvm-4 /opt/llvm-4.0

# FIXME(EricWF): Remove this hack once zorg has been updated.
RUN ln -s /opt/gcc-5/bin/gcc /usr/local/bin/gcc-4.9 && \
    ln -s /opt/gcc-5/bin/g++ /usr/local/bin/g++-4.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    buildbot-slave \
  && rm -rf /var/lib/apt/lists/*

ADD scripts/install_clang_packages.sh /tmp/
RUN /tmp/install_clang_packages.sh && rm /tmp/install_clang_packages.sh

RUN rm -rf /llvm-project/ && git clone --depth=1 https://github.com/llvm/llvm-project.git /llvm-project
