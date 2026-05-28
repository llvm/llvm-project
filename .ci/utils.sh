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
  sccache --show-stats
  sccache --show-stats >> artifacts/sccache_stats.txt
  cp "${MONOREPO_ROOT}"/*.ninja_log artifacts/ || :
  cp "${MONOREPO_ROOT}"/*.log artifacts/ || :
  cp "${BUILD_DIR}"/test-results.*.xml artifacts/ || :

  # Collect lldb per-test session and host logs (from `--channel "lldb event"`)
  # so we can post-mortem Windows DLL-load resolution in containers.
  # Wrapped in `set +e ... set -e` because a failure here must not skip the
  # GitHub Actions reporting block below.
  set +e
  shopt -s globstar nullglob
  lldb_test_log_root="${BUILD_DIR}/lldb-test-build.noindex"
  debug_log="artifacts/at-exit-debug.txt"
  {
    echo "=== at-exit lldb-test-logs collection ==="
    echo "PWD: $(pwd)"
    echo "BUILD_DIR: ${BUILD_DIR}"
    echo "lldb_test_log_root: ${lldb_test_log_root}"
    echo "exists: $([[ -d "${lldb_test_log_root}" ]] && echo yes || echo no)"
  } > "${debug_log}" 2>&1

  if [[ -d "${lldb_test_log_root}" ]]; then
    mkdir -p artifacts/lldb-test-logs
    # Use bash globstar instead of `find` because on the Windows GH Actions
    # runner, PATH resolves `find` to C:\Windows\System32\find.exe (DOS
    # string-search utility), which silently fails for filesystem traversal.
    matched=0
    for f in \
      "${lldb_test_log_root}"/**/Failure_*.log \
      "${lldb_test_log_root}"/**/*-host.log \
      "${lldb_test_log_root}"/**/ExpectedFailure_*.log \
      "${lldb_test_log_root}"/**/UnexpectedSuccess_*.log; do
      [[ -f "${f}" ]] || continue
      rel="${f#${lldb_test_log_root}/}"
      mkdir -p "artifacts/lldb-test-logs/$(dirname "${rel}")" 2>/dev/null
      cp "${f}" "artifacts/lldb-test-logs/${rel}" 2>/dev/null
      matched=$((matched + 1))
    done
    echo "matched=${matched}" >> "${debug_log}"
    echo "artifacts/lldb-test-logs file count: $(/usr/bin/find artifacts/lldb-test-logs -type f 2>/dev/null | wc -l)" >> "${debug_log}"
  fi
  shopt -u globstar nullglob
  set -e

  # If building fails there will be no results files.
  shopt -s nullglob

  if [[ -n "$GITHUB_ACTIONS" ]]; then
    python "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py \
      $retcode "${BUILD_DIR}"/test-results.*.xml "${MONOREPO_ROOT}"/ninja*.log \
      >> $GITHUB_STEP_SUMMARY
    if [[ -n "$GITHUB_PR_NUMBER" ]]; then
      (python "${MONOREPO_ROOT}"/.ci/premerge_advisor_explain.py \
        $(git rev-parse HEAD~1) $retcode "${GITHUB_TOKEN}" \
        $GITHUB_PR_NUMBER "${BUILD_DIR}"/test-results.*.xml \
        "${MONOREPO_ROOT}"/ninja*.log)
      advisor_retcode=$?
    else
      advisor_retcode=$retcode
    fi
  fi

  if [[ "$retcode" != "0" ]]; then
    if [[ -n "$GITHUB_ACTIONS" ]]; then
      python "${MONOREPO_ROOT}"/.ci/premerge_advisor_upload.py \
        $(git rev-parse HEAD~1) $GITHUB_RUN_NUMBER \
        "${BUILD_DIR}"/test-results.*.xml "${MONOREPO_ROOT}"/ninja*.log
    else
      python "${MONOREPO_ROOT}"/.ci/premerge_advisor_upload.py \
        $(git rev-parse HEAD) $BUILDBOT_BUILDNUMBER \
        "${BUILD_DIR}"/test-results.*.xml "${MONOREPO_ROOT}"/ninja*.log
    fi
  fi

  if [[ -n "$GITHUB_ACTIONS" ]]; then
    exit $advisor_retcode
  fi
}
trap at-exit EXIT

function start-group {
  groupname=$1
  if [[ -n "$GITHUB_ACTIONS" ]]; then
    echo "::endgroup"
    echo "::group::$groupname"
  elif [[ -n "$POSTCOMMIT_CI" ]]; then
    echo "@@@$STEP@@@"
  else
    echo "Starting $groupname"
  fi
}

export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r "${MONOREPO_ROOT}"/.ci/all_requirements.txt

# The ARM64 builders run on AWS and don't have access to the GCS cache.
if [[ -n "$GITHUB_ACTIONS" ]] && [[ "$RUNNER_ARCH" != "ARM64" ]]; then
  python .ci/cache_lit_timing_files.py download
fi
