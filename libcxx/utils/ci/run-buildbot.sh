#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

set -ex

BUILDER="${1}"

args=()
args+=("-DLLVM_ENABLE_PROJECTS=libcxx;libunwind;libcxxabi")
args+=("-DLIBCXX_CXX_ABI=libcxxabi")

case "${BUILDER}" in
x86_64-ubuntu-cxx03)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported --param=std=c++03")
;;
x86_64-ubuntu-cxx11)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported --param=std=c++11")
;;
x86_64-ubuntu-cxx14)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported --param=std=c++14")
;;
x86_64-ubuntu-cxx17)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported --param=std=c++17")
;;
x86_64-ubuntu-cxx2a)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported --param=std=c++2a")
;;
x86_64-ubuntu-noexceptions)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
    args+=("-DLIBCXX_ENABLE_EXCEPTIONS=OFF")
    args+=("-DLIBCXXABI_ENABLE_EXCEPTIONS=OFF")
;;
x86_64-ubuntu-32bit)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
    args+=("-DLLVM_BUILD_32_BITS=ON")
;;
x86_64-ubuntu-gcc)
    export CC=gcc
    export CXX=g++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
;;
x86_64-ubuntu-asan)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_USE_SANITIZER=Address")
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
;;
x86_64-ubuntu-msan)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_USE_SANITIZER=MemoryWithOrigins")
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
;;
x86_64-ubuntu-tsan)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_USE_SANITIZER=Thread")
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
;;
x86_64-ubuntu-ubsan)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_USE_SANITIZER=Undefined")
    args+=("-DLIBCXX_ABI_UNSTABLE=ON")
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
;;
x86_64-ubuntu-with_llvm_unwinder)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
    args+=("-DLIBCXXABI_USE_LLVM_UNWINDER=ON")
;;
x86_64-ubuntu-singlethreaded)
    export CC=clang
    export CXX=clang++
    args+=("-DLLVM_LIT_ARGS=-sv --show-unsupported")
    args+=("-DLIBCXX_ENABLE_THREADS=OFF")
    args+=("-DLIBCXXABI_ENABLE_THREADS=OFF")
    args+=("-DLIBCXX_ENABLE_MONOTONIC_CLOCK=OFF")
;;
*)
    echo "${BUILDER} is not a known configuration"
    exit 1
;;
esac

UMBRELLA_ROOT="$(git rev-parse --show-toplevel)"
LLVM_ROOT="${UMBRELLA_ROOT}/llvm"
BUILD_DIR="${UMBRELLA_ROOT}/build/${BUILDER}"

echo "--- Generating CMake"
rm -rf "${BUILD_DIR}"
cmake -S "${LLVM_ROOT}" -B "${BUILD_DIR}" -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo "${args[@]}"

echo "--- Building libc++ and libc++abi"
ninja -C "${BUILD_DIR}" check-cxx-deps cxxabi

echo "+++ Running the libc++ tests"
ninja -C "${BUILD_DIR}" check-cxx

echo "+++ Running the libc++abi tests"
ninja -C "${BUILD_DIR}" check-cxxabi

echo "+++ Running the libc++ benchmarks"
ninja -C "${BUILD_DIR}" check-cxx-benchmarks
