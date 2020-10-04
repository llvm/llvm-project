
#===-------------------------------------------------------------------------------------------===//
# buildslave
#===-------------------------------------------------------------------------------------------===//
ARG gcc_tot
ARG llvm_tot

FROM ${gcc_tot} AS gcc-tot
FROM ${llvm_tot} AS llvm-tot

FROM debian:stretch AS base-image

ADD install-packages.sh /tmp/
RUN /tmp/install-packages.sh && rm /tmp/install-packages.sh

COPY --from=ericwf/gcc:5.5.0 /compiler /opt/gcc-5

FROM base-image as worker-image

COPY --from=gcc-tot /compiler /opt/gcc-tot
COPY --from=llvm-tot /compiler /opt/llvm-tot

ENV PATH /opt/llvm-tot/bin:$PATH

RUN clang++ --version && echo hello
RUN g++ --version


RUN /opt/gcc-tot/bin/g++ --version
RUN /opt/llvm-tot/bin/clang++ --version
RUN /opt/llvm-tot/bin/clang --version

# FIXME(EricWF): remove this once the buildbot's config doesn't clobber the path.
RUN ln -s /opt/llvm-tot/bin/clang /usr/local/bin/clang
RUN ln -s /opt/llvm-tot/bin/clang++ /usr/local/bin/clang++


ADD run_buildbot.sh /
CMD /run_buildbot.sh /run/secrets/buildbot-auth
