#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//

# Build GCC versions
FROM ericwf/llvm-builder-base:latest
LABEL maintainer "libc++ Developers"

ARG install_prefix
ARG branch

ADD scripts/build_llvm_version.sh /tmp/
RUN /tmp/build_llvm_version.sh --install "$install_prefix" --branch "$branch" \
  && rm /tmp/build_llvm_version.sh
