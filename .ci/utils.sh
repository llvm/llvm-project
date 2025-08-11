#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# This script performs some setup and contains some utilities used for in the
# monolithic-linux.sh and monolithic-windows.sh scripts.

set -ex
set -o pipefail

MONOREPO_ROOT="${MONOREPO_ROOT:="$(git rev-parse --show-toplevel)"}"
BUILD_DIR="${BUILD_DIR:=${MONOREPO_ROOT}/build}"

rm -rf "${BUILD_DIR}"

sccache --zero-stats

function at-exit {
  retcode=$?

  mkdir -p artifacts
  sccache --show-stats >> artifacts/sccache_stats.txt
  cp "${BUILD_DIR}"/.ninja_log artifacts/.ninja_log
  cp "${BUILD_DIR}"/test-results.*.xml artifacts/ || :

  # If building fails there will be no results files.
  shopt -s nullglob

  if [[ "$GITHUB_STEP_SUMMARY" != "" ]]; then
    python "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py \
      $retcode "${BUILD_DIR}"/test-results.*.xml "${BUILD_DIR}"/ninja*.log \
      >> $GITHUB_STEP_SUMMARY
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
