#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This file contains functions to compute which projects should be built by CI
# systems and is intended to provide common functionality applicable across
# multiple systems during a transition period.
#

function compute-projects-to-test() {
  isForWindows=$1
  shift
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
      for p in bolt clang clang-tools-extra lld lldb mlir polly; do
        echo $p
      done
      # Flang is not stable in Windows CI at the moment
      if [[ $isForWindows == 0 ]]; then
        echo flang
      fi
    ;;
    clang)
      # lldb is temporarily removed to alleviate Linux pre-commit CI waiting times
      for p in clang-tools-extra compiler-rt cross-project-tests; do
        echo $p
      done
    ;;
    clang-tools-extra)
      echo libc
    ;;
    mlir)
      # Flang is not stable in Windows CI at the moment
      if [[ $isForWindows == 0 ]]; then
        echo flang
      fi
    ;;
    *)
      # Nothing to do
    ;;
    esac
  done
}

function compute-runtimes-to-test() {
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    clang)
      for p in libcxx libcxxabi libunwind; do
        echo $p
      done
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
      for p in clang lld llvm; do
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
    flang|lldb|libclc)
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
    lldb)                ;; # custom environment requirements (https://github.com/llvm/llvm-project/pull/94208#issuecomment-2146256857)
    bolt)                ;; # tests are not supported yet
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
  # Do not use "check-all" here because if there is "check-all" plus a
  # project specific target like "check-clang", that project's tests
  # will be run twice.
  projects=${@}
  for project in ${projects}; do
    case ${project} in
    clang-tools-extra)
      echo "check-clang-tools"
    ;;
    compiler-rt)
      echo "check-compiler-rt"
    ;;
    cross-project-tests)
      echo "check-cross-project"
    ;;
    libcxx)
      echo "check-cxx"
    ;;
    libcxxabi)
      echo "check-cxxabi"
    ;;
    libunwind)
      echo "check-unwind"
    ;;
    lldb)
      echo "check-lldb"
    ;;
    pstl)
      # Currently we do not run pstl tests in CI.
    ;;
    libclc)
      # Currently there is no testing for libclc.
    ;;
    *)
      echo "check-${project}"
    ;;
    esac
  done
}

