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
# most projects on macOS.
#

source .ci/utils.sh

projects="${1}"
targets="${2}"
runtimes="${3}"
runtime_targets="${4}"

start-group "CMake"

xcrun cmake -G Ninja \
      -B "${BUILD_DIR}" \
      -S "${MONOREPO_ROOT}"/llvm \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
      -D LLVM_DISABLE_ASSEMBLY_FILES=ON \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLDB_INCLUDE_TESTS=OFF \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D Python3_EXECUTABLE="${RUNNER_TEMP}/venv/bin/python3" \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests --succinct" \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache

start-group "ninja"

if [[ -n "${targets}" ]]; then
  ninja -C "${BUILD_DIR}" ${targets} 2>&1 | tee ninja.log
  cp ${BUILD_DIR}/.ninja_log ninja.ninja_log
fi

if [[ -n "${runtime_targets}" ]]; then
  start-group "ninja Runtimes"

  ninja -C "${BUILD_DIR}" ${runtime_targets} 2>&1 | tee ninja_runtimes.log
  cp ${BUILD_DIR}/.ninja_log ninja_runtimes.ninja_log
fi
