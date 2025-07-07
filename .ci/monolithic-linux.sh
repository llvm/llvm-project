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
INSTALL_DIR="${BUILD_DIR}/install"
rm -rf "${BUILD_DIR}"

ccache --zero-stats

if [[ -n "${CLEAR_CACHE:-}" ]]; then
  echo "clearing cache"
  ccache --clear
fi

mkdir -p artifacts/reproducers

# Make sure any clang reproducers will end up as artifacts.
export CLANG_CRASH_DIAGNOSTICS_DIR=`realpath artifacts/reproducers`

function at-exit {
  retcode=$?

  ccache --print-stats > artifacts/ccache_stats.txt
  cp "${BUILD_DIR}"/.ninja_log artifacts/.ninja_log
  cp "${BUILD_DIR}"/test-results.*.xml artifacts/ || :

  # If building fails there will be no results files.
  shopt -s nullglob
  
  python3 "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py ":penguin: Linux x64 Test Results" \
    $retcode "${BUILD_DIR}"/test-results.*.xml >> $GITHUB_STEP_SUMMARY
}
trap at-exit EXIT

projects="${1}"
targets="${2}"
runtimes="${3}"
runtime_targets="${4}"
runtime_targets_needs_reconfig="${5}"

lit_args="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests"

echo "::group::cmake"
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r "${MONOREPO_ROOT}"/.ci/all_requirements.txt

# Set the system llvm-symbolizer as preferred.
export LLVM_SYMBOLIZER_PATH=`which llvm-symbolizer`
[[ ! -f "${LLVM_SYMBOLIZER_PATH}" ]] && echo "llvm-symbolizer not found!"

# Set up all runtimes either way. libcxx is a dependency of LLDB.
# It will not be built unless it is used.
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
      -G Ninja \
      -D CMAKE_PREFIX_PATH="${HOME}/.local" \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="${lit_args}" \
      -D LLVM_ENABLE_LLD=ON \
      -D CMAKE_CXX_FLAGS=-gmlt \
      -D LLVM_CCACHE_BUILD=ON \
      -D LIBCXX_CXX_ABI=libcxxabi \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D LLDB_ENABLE_PYTHON=ON \
      -D LLDB_ENFORCE_STRICT_TEST_REQUIREMENTS=ON \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

echo "::endgroup::"
echo "::group::ninja"

# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" -k 0 ${targets}

echo "::endgroup::"

if [[ "${runtime_targets}" != "" ]]; then
  echo "::group::ninja runtimes"

  ninja -C "${BUILD_DIR}" ${runtime_targets}

  echo "::endgroup::"
fi

# Compiling runtimes with just-built Clang and running their tests
# as an additional testing for Clang.
if [[ "${runtime_targets_needs_reconfig}" != "" ]]; then
  echo "::group::cmake runtimes C++26"

  cmake \
    -D LIBCXX_TEST_PARAMS="std=c++26" \
    -D LIBCXXABI_TEST_PARAMS="std=c++26" \
    "${BUILD_DIR}"

  echo "::endgroup::"
  echo "::group::ninja runtimes C++26"

  ninja -C "${BUILD_DIR}" ${runtime_targets_needs_reconfig}

  echo "::endgroup::"
  echo "::group::cmake runtimes clang modules"

  cmake \
    -D LIBCXX_TEST_PARAMS="enable_modules=clang" \
    -D LIBCXXABI_TEST_PARAMS="enable_modules=clang" \
    "${BUILD_DIR}"

  echo "::endgroup::"
  echo "::group::ninja runtimes clang modules"

  ninja -C "${BUILD_DIR}" ${runtime_targets_needs_reconfig}

  echo "::endgroup::"
fi
