#!/usr/bin/env bash

set -eu
set -o pipefail

SCRIPTDIR=$(realpath $(dirname $0))
cd "${SCRIPTDIR}/.."

# Environment variables script works with:

# Set by buildkite
BUILDKITE_PULL_REQUEST_BASE_BRANCH=main^^
BUILDKITE_COMMIT=main
BUILDKITE_BRANCH=test
# Fetch origin to have an up to date  merge base for the diff.
git fetch origin
# List of files affected by this commit
: ${MODIFIED_FILES:=$(git diff --name-only origin/${BUILDKITE_PULL_REQUEST_BASE_BRANCH}...HEAD)}
# Filter rules for generic windows tests
: ${WINDOWS_AGENTS:='{"queue": "windows"}'}
# Filter rules for generic linux tests
: ${LINUX_AGENTS:='{"queue": "linux"}'}

reviewID="$(git log --format=%B -n 1 | sed -nE 's/^Review-ID:[[:space:]]*(.+)$/\1/p')"
if [[ "${reviewID}" != "" ]]; then
  buildMessage="https://llvm.org/${reviewID}"
else
  buildMessage="Push to branch ${BUILDKITE_BRANCH}"
fi

cat <<EOF
steps:
EOF

echo "Files modified:" >&2
echo "$MODIFIED_FILES" >&2
modified_dirs=$(echo "$MODIFIED_FILES" | cut -d'/' -f1 | sort -u)
echo "Directories modified:" >&2
echo "$modified_dirs" >&2

. ./.ci/compute-projects.sh

# Project specific pipelines.

# If libc++ or one of the runtimes directories changed.
if echo "$modified_dirs" | grep -q -E "^(libcxx|libcxxabi|libunwind|runtimes|cmake)$"; then
  cat <<EOF
- trigger: "libcxx-ci"
  build:
    message: "${buildMessage}"
    commit: "${BUILDKITE_COMMIT}"
    branch: "${BUILDKITE_BRANCH}"
EOF
fi

# Generic pipeline for projects that have not defined custom steps.
#
# Individual projects should instead define the pre-commit CI tests that suits their
# needs while letting them run on the infrastructure provided by LLVM.

# Figure out which projects need to be built on each platform
all_projects="bolt clang clang-tools-extra compiler-rt cross-project-tests flang flang-rt libc libclc lld lldb llvm mlir openmp polly pstl"
modified_projects="$(keep-modified-projects ${all_projects})"

linux_projects_to_test=$(exclude-linux $(compute-projects-to-test 0 ${modified_projects}))
linux_check_targets=$(check-targets ${linux_projects_to_test} | sort | uniq)
linux_projects=$(add-dependencies ${linux_projects_to_test} | sort | uniq)

linux_runtimes_to_test=$(compute-runtimes-to-test ${linux_projects_to_test})
linux_runtime_check_targets=$(check-targets ${linux_runtimes_to_test} | sort | uniq)
linux_runtimes=$(echo ${linux_runtimes_to_test} | sort | uniq)

windows_projects_to_test=$(exclude-windows $(compute-projects-to-test 1 ${modified_projects}))
windows_check_targets=$(check-targets ${windows_projects_to_test} | sort | uniq)
windows_projects=$(add-dependencies ${windows_projects_to_test} | sort | uniq)

# Generate the appropriate pipeline
if [[ "${linux_projects}" != "" ]]; then
  cat <<EOF
- label: ':linux: Linux x64'
  artifact_paths:
  - 'artifacts/**/*'
  - '*_result.json'
  - 'build/test-results.*.xml'
  agents: ${LINUX_AGENTS}
  retry:
    automatic:
      - exit_status: -1  # Agent was lost
        limit: 2
      - exit_status: 255 # Forced agent shutdown
        limit: 2
  timeout_in_minutes: 120
  env:
    CC: 'clang'
    CXX: 'clang++'
  commands:
  - './.ci/monolithic-linux.sh "$(echo ${linux_projects} | tr ' ' ';')" "$(echo ${linux_check_targets})" "$(echo ${linux_runtimes} | tr ' ' ';')" "$(echo ${linux_runtime_check_targets})"'
EOF
fi

if [[ "${windows_projects}" != "" ]]; then
  cat <<EOF
- label: ':windows: Windows x64'
  artifact_paths:
  - 'artifacts/**/*'
  - '*_result.json'
  - 'build/test-results.*.xml'
  agents: ${WINDOWS_AGENTS}
  retry:
    automatic:
      - exit_status: -1  # Agent was lost
        limit: 2
      - exit_status: 255 # Forced agent shutdown
        limit: 2
  timeout_in_minutes: 150
  env:
    MAX_PARALLEL_COMPILE_JOBS: '16'
    MAX_PARALLEL_LINK_JOBS: '4'
  commands:
  - 'C:\\BuildTools\\Common7\\Tools\\VsDevCmd.bat -arch=amd64 -host_arch=amd64'
  - 'bash .ci/monolithic-windows.sh "$(echo ${windows_projects} | tr ' ' ';')" "$(echo ${windows_check_targets})"'
EOF
fi

echo ./.ci/monolithic-local.sh "$(echo ${linux_projects} | tr ' ' ';')" "$(echo ${linux_check_targets})" "$(echo ${linux_runtimes} | tr ' ' ';')" "$(echo ${linux_runtime_check_targets})"'
./.ci/monolithic-local.sh "$(echo ${linux_projects} | tr ' ' ';')" "$(echo ${linux_check_targets})" "check-flang"  "check-flang-rt"'
#./.ci/monolithic-local.sh "bolt;clang;clang-tools-extra;flang;lld;lldb;llvm;mlir;polly" "check-bolt check-clang check-clang-tools check-flang check-lld check-lldb check-llvm check-mlir check-polly" "libcxx;libcxxabi;libunwind;flang-rt" "check-cxx check-cxxabi check-unwind check-flang-rt"
