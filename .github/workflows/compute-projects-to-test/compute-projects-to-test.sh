#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This file generates a Buildkite pipeline that triggers the various CI jobs for
# the LLVM project during pre-commit CI.
#
# See https://buildkite.com/docs/agent/v3/cli-pipeline#pipeline-format.
#
# As this outputs a yaml file, it's possible to log messages to stderr or
# prefix with "#".


set -eu
set -o pipefail

# Environment variables script works with:

# Set by GitHub
: ${GITHUB_OUTPUT:=}
: ${RUNNER_OS:=}

# Allow users to specify which projects to build.
all_projects="bolt clang clang-tools-extra compiler-rt cross-project-tests flang libc libclc lld lldb llvm mlir openmp polly pstl"
if [ "$#" -ne 0 ]; then
  wanted_projects="${@}"
else
  wanted_projects="${all_projects}"
fi

# List of files affected by this commit
: ${MODIFIED_FILES:=$(git diff --name-only HEAD~1...HEAD)}

echo "Files modified:" >&2
echo "$MODIFIED_FILES" >&2
modified_dirs=$(echo "$MODIFIED_FILES" | cut -d'/' -f1 | sort -u)
echo "Directories modified:" >&2
echo "$modified_dirs" >&2
echo "wanted_projects: $wanted_projects"

function remove-unwanted-projects() {
  projects=${@}
  for project in ${projects}; do
    if echo "$wanted_projects" | tr ' ' '\n' | grep -q -E "^${project}$"; then
      echo "${project}"
    fi
  done
}

function compute-projects-to-test() {
  projects=${@}
  for project in ${projects}; do
    echo "${project}"
    case ${project} in
    lld)
      for p in bolt cross-project-tests; do
        echo $p
      done
    ;;
    llvm)
      for p in bolt clang clang-tools-extra flang lld lldb mlir polly; do
        echo $p
      done
    ;;
    clang)
      for p in clang-tools-extra compiler-rt flang libc lldb openmp cross-project-tests; do
        echo $p
      done
    ;;
    clang-tools-extra)
      echo libc
    ;;
    mlir)
      echo flang
    ;;
    *)
      # Nothing to do
    ;;
    esac
  done
}

function add-dependencies() {
  projects=${@}
  for project in ${projects}; do
    echo "${project}"
    case ${project} in
    bolt)
      for p in lld llvm; do
        echo $p
      done
    ;;
    cross-project-tests)
      for p in lld clang; do
        echo $p
      done
    ;;
    clang-tools-extra)
      for p in llvm clang; do
        echo $p
      done
    ;;
    compiler-rt|libc|openmp)
      echo clang lld
    ;;
    flang|lldb)
      for p in llvm clang; do
        echo $p
      done
    ;;
    lld|mlir|polly)
      echo llvm
    ;;
    *)
      # Nothing to do
    ;;
    esac
  done
}

function exclude-linux() {
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    cross-project-tests) ;; # tests failing
    lldb)                ;; # tests failing
    openmp)              ;; # https://github.com/google/llvm-premerge-checks/issues/410
    *)
      echo "${project}"
    ;;
    esac
  done
}

function exclude-windows() {
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    cross-project-tests) ;; # tests failing
    compiler-rt)         ;; # tests taking too long
    openmp)              ;; # TODO: having trouble with the Perl installation
    libc)                ;; # no Windows support
    lldb)                ;; # tests failing
    bolt)                ;; # tests are not supported yet
    *)
      echo "${project}"
    ;;
    esac
  done
}

function exclude-mac() {
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    cross-project-tests) ;; # tests failing
    openmp)              ;; # https://github.com/google/llvm-premerge-checks/issues/410
    lldb)                ;; # tests failing
    flang)               ;; # tests failing
    bolt)                ;; # tests failing
    *)
      echo "${project}"
    ;;
    esac
  done

}

# Prints only projects that are both present in $modified_dirs and the passed
# list.
function keep-modified-projects() {
  projects=${@}
  for project in ${projects}; do
    if echo "$modified_dirs" | grep -q -E "^${project}$"; then
      echo "${project}"
    fi
  done
}

function check-targets() {
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    clang-tools-extra)
      echo "check-clang-tools"
    ;;
    compiler-rt)
      echo "check-all"
    ;;
    cross-project-tests)
      echo "check-cross-project"
    ;;
    lldb)
      echo "check-all" # TODO: check-lldb may not include all the LLDB tests?
    ;;
    pstl)
      echo "check-all"
    ;;
    libclc)
      echo "check-all"
    ;;
    *)
      echo "check-${project}"
    ;;
    esac
  done
}

# Generic pipeline for projects that have not defined custom steps.
#
# Individual projects should instead define the pre-commit CI tests that suits their
# needs while letting them run on the infrastructure provided by LLVM.

# Figure out which projects need to be built on each platform
modified_projects="$(keep-modified-projects ${all_projects})"
echo "modified_projects: $modified_projects"

if [ "${RUNNER_OS}" = "Linux" ]; then
  projects_to_test=$(exclude-linux $(compute-projects-to-test ${modified_projects}))
elif [ "${RUNNER_OS}" = "Windows" ]; then
  projects_to_test=$(exclude-windows $(compute-projects-to-test ${modified_projects}))
elif [ "${RUNNER_OS}" = "macOS" ]; then
  projects_to_test=$(exclude-mac $(compute-projects-to-test ${modified_projects}))
else
  echo "Unknown runner OS: $RUNNER_OS"
  exit 1
fi
check_targets=$(check-targets $(remove-unwanted-projects ${projects_to_test}) | sort | uniq)
projects=$(remove-unwanted-projects $(add-dependencies ${projects_to_test}) | sort | uniq)

echo "$RUNNER_OS-check-targets=$(echo ${check_targets} | tr ' ' ' ')" >> $GITHUB_OUTPUT
echo "$RUNNER_OS-projects=$(echo ${projects} | tr ' ' ';')" >> $GITHUB_OUTPUT

cat $GITHUB_OUTPUT
