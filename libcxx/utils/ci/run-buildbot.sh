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
MONOREPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="${MONOREPO_ROOT}/build/${BUILDER}"
INSTALL_DIR="${MONOREPO_ROOT}/build/${BUILDER}/install"

function generate-cmake() {
    echo "--- Generating CMake"
    rm -rf "${BUILD_DIR}"
    cmake -S "${MONOREPO_ROOT}/llvm" \
          -B "${BUILD_DIR}" \
          -GNinja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DLLVM_ENABLE_PROJECTS="libcxx;libunwind;libcxxabi" \
          -DLLVM_LIT_ARGS="-sv --show-unsupported --xunit-xml-output test-results.xml" \
          -DLIBCXX_CXX_ABI=libcxxabi \
          ${@}
}

function check-cxx-cxxabi() {
    echo "+++ Running the libc++ tests"
    ninja -C "${BUILD_DIR}" check-cxx

    echo "+++ Running the libc++abi tests"
    ninja -C "${BUILD_DIR}" check-cxxabi

    echo "--- Installing libc++ and libc++abi to a fake location"
    ninja -C "${BUILD_DIR}" install-cxx install-cxxabi
}

function check-cxx-benchmarks() {
    echo "--- Running the benchmarks"
    ninja -C "${BUILD_DIR}" check-cxx-benchmarks
}

case "${BUILDER}" in
generic-cxx03)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-cxx03.cmake"
    check-cxx-cxxabi
;;
generic-cxx11)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-cxx11.cmake"
    check-cxx-cxxabi
;;
generic-cxx14)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-cxx14.cmake"
    check-cxx-cxxabi
;;
generic-cxx17)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-cxx17.cmake"
    check-cxx-cxxabi
;;
generic-cxx2a)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-cxx2a.cmake"
    check-cxx-cxxabi
;;
generic-noexceptions)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-noexceptions.cmake"
    check-cxx-cxxabi
;;
generic-32bit)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-32bits.cmake"
    check-cxx-cxxabi
;;
generic-gcc)
    export CC=gcc
    export CXX=g++
    # FIXME: Re-enable experimental testing on GCC. GCC cares about the order
    #        in which we link -lc++experimental, which causes issues.
    generate-cmake -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=OFF
    check-cxx-cxxabi
;;
generic-asan)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-asan.cmake"
    check-cxx-cxxabi
;;
generic-msan)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-msan.cmake"
    check-cxx-cxxabi
;;
generic-tsan)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-tsan.cmake"
    check-cxx-cxxabi
;;
generic-ubsan)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-ubsan.cmake"
    check-cxx-cxxabi
;;
generic-with_llvm_unwinder)
    export CC=clang
    export CXX=clang++
    generate-cmake -DLIBCXXABI_USE_LLVM_UNWINDER=ON
    check-cxx-cxxabi
;;
generic-singlethreaded)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-singlethreaded.cmake"
    check-cxx-cxxabi
;;
generic-nodebug)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-nodebug.cmake"
    check-cxx-cxxabi
;;
generic-no-random_device)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-no-random_device.cmake"
    check-cxx-cxxabi
;;
generic-no-localization)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Generic-no-localization.cmake"
    check-cxx-cxxabi
;;
x86_64-apple-system)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Apple.cmake"
    check-cxx-cxxabi
;;
x86_64-apple-system-noexceptions)
    export CC=clang
    export CXX=clang++
    generate-cmake -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Apple.cmake" \
                   -DLIBCXX_ENABLE_EXCEPTIONS=OFF \
                   -DLIBCXXABI_ENABLE_EXCEPTIONS=OFF
    check-cxx-cxxabi
;;
benchmarks)
    export CC=clang
    export CXX=clang++
    generate-cmake
    check-cxx-benchmarks
;;
unified-standalone)
    export CC=clang
    export CXX=clang++

    echo "--- Generating CMake"
    rm -rf "${BUILD_DIR}"
    cmake -S "${MONOREPO_ROOT}/libcxx/utils/ci/runtimes" \
          -B "${BUILD_DIR}" \
          -GNinja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi;libunwind"

    check-cxx-cxxabi
;;
legacy-standalone)
    export CC=clang
    export CXX=clang++

    echo "--- Generating CMake"
    rm -rf "${BUILD_DIR}"
    cmake -S "${MONOREPO_ROOT}/libcxx" \
          -B "${BUILD_DIR}/libcxx" \
          -GNinja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DLLVM_PATH="${MONOREPO_ROOT}/llvm" \
          -DLIBCXX_CXX_ABI=libcxxabi \
          -DLIBCXX_CXX_ABI_INCLUDE_PATHS="${MONOREPO_ROOT}/libcxxabi/include" \
          -DLIBCXX_CXX_ABI_LIBRARY_PATH="${BUILD_DIR}/libcxxabi/lib"

    cmake -S "${MONOREPO_ROOT}/libcxxabi" \
          -B "${BUILD_DIR}/libcxxabi" \
          -GNinja \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DLLVM_PATH="${MONOREPO_ROOT}/llvm" \
          -DLIBCXXABI_LIBCXX_PATH="${MONOREPO_ROOT}/libcxx" \
          -DLIBCXXABI_LIBCXX_INCLUDES="${MONOREPO_ROOT}/libcxx/include" \
          -DLIBCXXABI_LIBCXX_LIBRARY_PATH="${BUILD_DIR}/libcxx/lib"

    echo "+++ Building libc++abi"
    ninja -C "${BUILD_DIR}/libcxxabi" cxxabi

    echo "+++ Building libc++"
    ninja -C "${BUILD_DIR}/libcxx" cxx

    echo "+++ Running the libc++ tests"
    ninja -C "${BUILD_DIR}/libcxx" check-cxx

    echo "+++ Running the libc++abi tests"
    ninja -C "${BUILD_DIR}/libcxxabi" check-cxxabi
;;
*)
    echo "${BUILDER} is not a known configuration"
    exit 1
;;
esac
