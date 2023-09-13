#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This script performs a monolithic build of the monorepo and runs the tests of
# most projects on Linux. This should be replaced by per-project scripts that
# run only the relevant tests.
#

set -ex
set -o pipefail

MONOREPO_ROOT="${MONOREPO_ROOT:="$(git rev-parse --show-toplevel)"}"
BUILD_DIR="${BUILD_DIR:=${MONOREPO_ROOT}/build}"
rm -rf ${BUILD_DIR}

ccache --zero-stats

if [[ -n "${CLEAR_CACHE:-}" ]]; then
  echo "clearing cache"
  ccache --clear
fi

function show-stats {
  mkdir -p artifacts
  ccache --print-stats > artifacts/ccache_stats.txt
}
trap show-stats EXIT

projects="${1}"
targets="${2}"

echo "--- cmake"
pip install -q -r ${MONOREPO_ROOT}/mlir/python/requirements.txt
cmake -S ${MONOREPO_ROOT}/llvm -B ${BUILD_DIR} \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml" \
      -D LLVM_ENABLE_LLD=ON \
      -D CMAKE_CXX_FLAGS=-gmlt \
      -D BOLT_CLANG_EXE=/usr/bin/clang \
      -D LLVM_CCACHE_BUILD=ON

echo "--- ninja"
# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" ${targets}
