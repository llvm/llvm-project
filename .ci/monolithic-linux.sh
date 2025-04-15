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

  # If building fails there will be no results files.
  shopt -s nullglob
  if command -v buildkite-agent 2>&1 >/dev/null
  then
    python3 "${MONOREPO_ROOT}"/.ci/generate_test_report_buildkite.py ":linux: Linux x64 Test Results" \
      "linux-x64-test-results" $retcode "${BUILD_DIR}"/test-results.*.xml
  else
    python3 "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py ":linux: Linux x64 Test Results" \
      $retcode "${BUILD_DIR}"/test-results.*.xml >> $GITHUB_STEP_SUMMARY
  fi
}
trap at-exit EXIT

projects="${1}"
targets="${2}"
runtimes="${3}"

lit_args="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests"

echo "--- cmake"

export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r "${MONOREPO_ROOT}"/mlir/python/requirements.txt
pip install -q -r "${MONOREPO_ROOT}"/lldb/test/requirements.txt
pip install -q -r "${MONOREPO_ROOT}"/.ci/requirements.txt

# Set the system llvm-symbolizer as preferred.
export LLVM_SYMBOLIZER_PATH=`which llvm-symbolizer`
[[ ! -f "${LLVM_SYMBOLIZER_PATH}" ]] && echo "llvm-symbolizer not found!"

# Set up all runtimes either way. libcxx is a dependency of LLDB.
# If it ends up being unused, not much harm.
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -D LLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
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
      -D LLDB_ENFORCE_STRICT_TEST_REQUIREMENTS=ON

echo "--- ninja"

# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" -k 0 ${targets}

runtime_targets="${4}"

# Run runtimes tests.
# We don't need to do a clean separate build of runtimes, because runtimes
# will be built against just built clang, and because LIBCXX_TEST_PARAMS
# and LIBCXXABI_TEST_PARAMS only affect lit configuration, which successfully
# propagates without a clean build. Other that those two variables, builds
# are supposed to be the same.
if [[ "${runtimes}" != "" ]]; then
  if [[ "${runtime_targets}" == "" ]]; then
    echo "Runtimes to build are specified, but targets are not."
    exit 1
  fi

  echo "--- cmake runtimes C++26"

  cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LIBCXX_TEST_PARAMS="std=c++26" \
      -D LIBCXXABI_TEST_PARAMS="std=c++26"

  echo "--- ninja runtimes C++26"

  ninja -vC "${BUILD_DIR}" ${runtime_targets}

  echo "--- cmake runtimes clang modules"

  cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LIBCXX_TEST_PARAMS="enable_modules=clang" \
      -D LIBCXXABI_TEST_PARAMS="enable_modules=clang"

  echo "--- ninja runtimes clang modules"

  ninja -vC "${BUILD_DIR}" ${runtime_targets}
fi
