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
# most projects on Windows. This should be replaced by per-project scripts that
# run only the relevant tests.
#

set -ex
set -o pipefail

MONOREPO_ROOT="${MONOREPO_ROOT:="$(git rev-parse --show-toplevel)"}"
BUILD_DIR="${BUILD_DIR:=${MONOREPO_ROOT}/build}"

rm -rf "${BUILD_DIR}"

if [[ -n "${CLEAR_CACHE:-}" ]]; then
  echo "clearing sccache"
  rm -rf "$SCCACHE_DIR"
fi

sccache --zero-stats
function at-exit {
  retcode=$?

  mkdir -p artifacts
  sccache --show-stats >> artifacts/sccache_stats.txt

  # If building fails there will be no results files.
  shopt -s nullglob
  if command -v buildkite-agent 2>&1 >/dev/null
  then
    python "${MONOREPO_ROOT}"/.ci/generate_test_report.py ":windows: Windows x64 Test Results" \
      "windows-x64-test-results" $retcode "${BUILD_DIR}"/test-results.*.xml
  fi
}
trap at-exit EXIT

projects="${1}"
targets="${2}"

echo "--- cmake"
pip install -q -r "${MONOREPO_ROOT}"/mlir/python/requirements.txt
pip install -q -r "${MONOREPO_ROOT}"/.ci/requirements.txt

export CC=cl
export CXX=cl
export LD=link

# The CMAKE_*_LINKER_FLAGS to disable the manifest come from research
# on fixing a build reliability issue on the build server, please
# see https://github.com/llvm/llvm-project/pull/82393 and
# https://discourse.llvm.org/t/rfc-future-of-windows-pre-commit-ci/76840/40
# for further information.
# We limit the number of parallel compile jobs to 24 control memory
# consumption and improve build reliability.
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests" \
      -D COMPILER_RT_BUILD_ORC=OFF \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_MODULE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO" \
      -D LLVM_PARALLEL_COMPILE_JOBS=${MAX_PARALLEL_COMPILE_JOBS} \
      -D LLVM_PARALLEL_LINK_JOBS=${MAX_PARALLEL_LINK_JOBS}

echo "--- ninja"
# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" -k 0 ${targets}
