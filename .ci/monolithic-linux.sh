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

sccache --zero-stats

mkdir -p artifacts/reproducers

# Make sure any clang reproducers will end up as artifacts.
export CLANG_CRASH_DIAGNOSTICS_DIR=`realpath artifacts/reproducers`

function at-exit {
  retcode=$?

  sccache --show-stats > artifacts/sccache_stats.txt
  cp "${BUILD_DIR}"/.ninja_log artifacts/.ninja_log
  cp "${BUILD_DIR}"/test-results.*.xml artifacts/ || :

  # If building fails there will be no results files.
  shopt -s nullglob
  
  if [[ "$GITHUB_STEP_SUMMARY" != "" ]]; then
    python3 "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py \
      $retcode "${BUILD_DIR}"/test-results.*.xml >> $GITHUB_STEP_SUMMARY
  fi
}
trap at-exit EXIT

function start-group {
  groupname=$1
  if [[ "$GITHUB_ACTIONS" != "" ]]; then
    echo "::endgroup"
    echo "::group::$groupname"
  elif [[ "$POSTCOMMIT_CI" != "" ]]; then
    echo "@@@$STEP@@@"
  else
    echo "Starting $groupname"
  fi
}

projects="${1}"
targets="${2}"
runtimes="${3}"
runtime_targets="${4}"
runtime_targets_needs_reconfig="${5}"
enable_cir="${6}"

lit_args="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests"

start-group "CMake"
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
      -D CLANG_ENABLE_CIR=${enable_cir} \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="${lit_args}" \
      -D LLVM_ENABLE_LLD=ON \
      -D CMAKE_CXX_FLAGS=-gmlt \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -D LIBCXX_CXX_ABI=libcxxabi \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D LLDB_ENABLE_PYTHON=ON \
      -D LLDB_ENFORCE_STRICT_TEST_REQUIREMENTS=ON \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

start-group "ninja"

# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" -k 0 ${targets}

if [[ "${runtime_targets}" != "" ]]; then
  start-group "ninja Runtimes"

  ninja -C "${BUILD_DIR}" ${runtime_targets}
fi

# Compiling runtimes with just-built Clang and running their tests
# as an additional testing for Clang.
if [[ "${runtime_targets_needs_reconfig}" != "" ]]; then
  start-group "CMake Runtimes C++26"

  cmake \
    -D LIBCXX_TEST_PARAMS="std=c++26" \
    -D LIBCXXABI_TEST_PARAMS="std=c++26" \
    "${BUILD_DIR}"

  start-group "ninja Runtimes C++26"

  ninja -C "${BUILD_DIR}" ${runtime_targets_needs_reconfig}

  start-group "CMake Runtimes Clang Modules"

  cmake \
    -D LIBCXX_TEST_PARAMS="enable_modules=clang" \
    -D LIBCXXABI_TEST_PARAMS="enable_modules=clang" \
    "${BUILD_DIR}"

  start-group "ninja Runtimes Clang Modules"

  ninja -C "${BUILD_DIR}" ${runtime_targets_needs_reconfig}
fi
