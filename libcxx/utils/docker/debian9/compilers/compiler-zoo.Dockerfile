#===- libcxx/utils/docker/debian9/Dockerfile --------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-------------------------------------------------------------------------------------------===//

#===-------------------------------------------------------------------------------------------===//
#  compiler-zoo
#===-------------------------------------------------------------------------------------------===//
FROM  ericwf/llvm-builder-base:latest AS compiler-zoo
LABEL maintainer "libc++ Developers"

# Copy over the GCC and Clang installations
COPY --from=ericwf/compiler:gcc-4.8.5 /opt/gcc-4.8.5 /opt/gcc-4.8.5
COPY --from=ericwf/compiler:gcc-4.9.4 /opt/gcc-4.9.4 /opt/gcc-4.9.4
COPY --from=ericwf/compiler:gcc-5 /opt/gcc-5   /opt/gcc-5
COPY --from=ericwf/compiler:gcc-6 /opt/gcc-6   /opt/gcc-6
COPY --from=ericwf/compiler:gcc-7 /opt/gcc-7   /opt/gcc-7
COPY --from=ericwf/compiler:gcc-8 /opt/gcc-8   /opt/gcc-8
COPY --from=ericwf/compiler:gcc-tot /opt/gcc-tot /opt/gcc-tot

COPY --from=ericwf/compiler:llvm-3.6 /opt/llvm-3.6 /opt/llvm-3.6
COPY --from=ericwf/compiler:llvm-3.7 /opt/llvm-3.7 /opt/llvm-3.7
COPY --from=ericwf/compiler:llvm-3.8 /opt/llvm-3.8 /opt/llvm-3.8
COPY --from=ericwf/compiler:llvm-3.9 /opt/llvm-3.9 /opt/llvm-3.9
COPY --from=ericwf/compiler:llvm-4 /opt/llvm-4 /opt/llvm-4
COPY --from=ericwf/compiler:llvm-5 /opt/llvm-5 /opt/llvm-5
COPY --from=ericwf/compiler:llvm-6 /opt/llvm-6 /opt/llvm-6
COPY --from=ericwf/compiler:llvm-7 /opt/llvm-7 /opt/llvm-7
COPY --from=ericwf/compiler:llvm-8 /opt/llvm-8 /opt/llvm-8
COPY --from=ericwf/compiler:llvm-tot /opt/llvm-tot /opt/llvm-tot


