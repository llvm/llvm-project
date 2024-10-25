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

function at-exit {
  python3 "${MONOREPO_ROOT}"/.ci/generate_test_report.py ":linux: Linux x64 Test Results" \
    "linux-x64-test-results" "${BUILD_DIR}"/*-test-results.xml

  mkdir -p artifacts
  ccache --print-stats > artifacts/ccache_stats.txt
}
trap at-exit EXIT

# TODO: separate file for import into Windows script?
function ninja-targets {
  # $1 is the ninja arguments to use
  # $2 is is an optional postfix to add to the target name when renaming result files.
  # $3 is the list of targets
  set +e
  err_code=0
  for target in $3; do
    ninja $1 ${target}
    new_err_code=$?
    if [[ $new_err_code -ne 0 ]]; then
      err_code=${new_err_code}
    fi
    mv "${BUILD_DIR}/test-results.xml" "${BUILD_DIR}/${target}${2}-test-results.xml"
  done

  if [[ $err_code -ne 0 ]]; then
   exit $err_code
  fi

  set -e
}

projects="${1}"
targets="${2}"

lit_args="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests"

echo "--- cmake"
pip install -q -r "${MONOREPO_ROOT}"/mlir/python/requirements.txt
pip install -q -r "${MONOREPO_ROOT}"/lldb/test/requirements.txt
pip install -q junitparser==3.2.0
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="${lit_args}" \
      -D LLVM_ENABLE_LLD=ON \
      -D CMAKE_CXX_FLAGS=-gmlt \
      -D LLVM_CCACHE_BUILD=ON \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

echo "--- ninja"

ninja-targets "-C "${BUILD_DIR}" -k 0" "" "$targets"

runtimes="${3}"
runtime_targets="${4}"

# Compiling runtimes with just-built Clang and running their tests
# as an additional testing for Clang.
if [[ "${runtimes}" != "" ]]; then
  if [[ "${runtime_targets}" == "" ]]; then
    echo "Runtimes to build are specified, but targets are not."
    exit 1
  fi

  echo "--- ninja install-clang"

  ninja -C ${BUILD_DIR} install-clang install-clang-resource-headers

  RUNTIMES_BUILD_DIR="${MONOREPO_ROOT}/build-runtimes"
  INSTALL_DIR="${BUILD_DIR}/install"
  mkdir -p ${RUNTIMES_BUILD_DIR}

  echo "--- cmake runtimes C++03"

  cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
      -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
      -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
      -D LIBCXX_CXX_ABI=libcxxabi \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -D LIBCXX_TEST_PARAMS="std=c++03" \
      -D LIBCXXABI_TEST_PARAMS="std=c++03" \
      -D LLVM_LIT_ARGS="${lit_args}"

  echo "--- ninja runtimes C++03"

  # TODO: there's no way to tell a failure here apart from a failure of the same
  # test in the other build mode.
  ninja-targets "-vC "${RUNTIMES_BUILD_DIR}"" "-cxx03" "${runtime_targets}"

  echo "--- cmake runtimes C++26"

  rm -rf "${RUNTIMES_BUILD_DIR}"
  cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
      -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
      -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
      -D LIBCXX_CXX_ABI=libcxxabi \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -D LIBCXX_TEST_PARAMS="std=c++26" \
      -D LIBCXXABI_TEST_PARAMS="std=c++26" \
      -D LLVM_LIT_ARGS="${lit_args}"

  echo "--- ninja runtimes C++26"

  ninja-targets "-vC "${RUNTIMES_BUILD_DIR}"" "-cxx26" "${runtime_targets}"

  echo "--- cmake runtimes clang modules"

  rm -rf "${RUNTIMES_BUILD_DIR}"
  cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
      -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
      -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
      -D LIBCXX_CXX_ABI=libcxxabi \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -D LIBCXX_TEST_PARAMS="enable_modules=clang" \
      -D LIBCXXABI_TEST_PARAMS="enable_modules=clang" \
      -D LLVM_LIT_ARGS="${lit_args}"

  echo "--- ninja runtimes clang modules"

  ninja-targets "-vC "${RUNTIMES_BUILD_DIR}"" "-modules" "${runtime_targets}"
fi
