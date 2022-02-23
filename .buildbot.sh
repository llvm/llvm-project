#!/bin/sh

set -e

# Buildbot will have done a shallow clone, but the formatting step requires
# both full git history, and a complete set of upstream llvm release tags.
git fetch --unshallow
git fetch --tags https://github.com/llvm/llvm-project

INST_DIR=`pwd`/inst

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    ../llvm
make -j `nproc` install

# There are many test suites for LLVM:
# https://llvm.org/docs/TestingGuide.html
#
# This runs unit and integration tests.
make -j `nproc` check-all
cd ..

# clang-format any new files that we've introduced ourselves.
PATH=${INST_DIR}/bin:${PATH}
sh yk_format_new_files.sh
git diff --exit-code

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
