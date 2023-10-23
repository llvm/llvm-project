#!/bin/sh

set -e

INST_DIR=`pwd`/inst

mkdir -p build
cd build
# Disabling default PIE due to:
# https://github.com/llvm/llvm-project/issues/57085
cmake -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    -DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -GNinja \
    ../llvm
cmake --build .
cmake --install .

# clang-format any new files that we've introduced ourselves.
cd ..
PATH=${INST_DIR}/bin:${PATH}
git fetch origin main:refs/remotes/origin/main
sh yk_format_new_files.sh
git diff --exit-code

# There are many test suites for LLVM:
# https://llvm.org/docs/TestingGuide.html
#
# This runs unit and integration tests.
cmake --build build --target check-all

# FIXME The commented code below should run the `test-suite` tests, as
# described at https://llvm.org/docs/TestSuiteGuide.html.
#
# However, the suite fails (even on a stock LLVM) with this error:
#   retref-bench.cc:17:10: fatal error: 'xray/xray_interface.h' file not found
#   #include "xray/xray_interface.h"
#            ^~~~~~~~~~~~~~~~~~~~~~~
#
#git clone --depth 1 --branch llvmorg-12.0.0 https://github.com/llvm/llvm-test-suite.git test-suite
#mkdir -p test-suite-build
#cd test-suite-build
#cmake -DCMAKE_C_COMPILER=${INST_DIR}/bin/clang \
#	-C../test-suite/cmake/caches/O3.cmake \
#	../test-suite
#make -j `nproc`
#llvm-lit -v -j 1 -o results.json .

# Check ykllvm builds with assertions disabled.
#
# In the past things like `Value::dump()` have slipped in to the repo and these
# are not available if LLVM is built without assertions (you'd get a linker
# error).
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    -DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -GNinja \
    ../llvm
cmake --build .
