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
# most projects on Windows. This should be replaced by per-project scripts that
# run only the relevant tests.
#

source .ci/utils.sh

projects="${1}"
targets="${2}"
runtimes="${3}"
runtimes_targets="${4}"

start-group "CMake"
pip install -q -r "${MONOREPO_ROOT}"/.ci/all_requirements.txt


mkdir /tmp/xz-download
pushd /tmp/xz-download
curl -L -o xz-5.8.1-windows.zip http://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1-windows.zip
unzip xz-5.8.1-windows.zip
ls -l /tmp/xz-download/bin_x86-64/xz.exe
popd

mkdir /tmp/clang-download
pushd /tmp/clang-download
curl -L -o "clang+llvm-21.1.2-x86_64-pc-windows-msvc.tar.xz" http://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.2/clang+llvm-21.1.2-x86_64-pc-windows-msvc.tar.xz
ls -l "clang+llvm-21.1.2-x86_64-pc-windows-msvc.tar.xz"
/tmp/xz-download/bin_x86-64/xz.exe -d -qq "clang+llvm-21.1.2-x86_64-pc-windows-msvc.tar.xz"
tar xf "clang+llvm-21.1.2-x86_64-pc-windows-msvc.tar"
ls -l /tmp/clang-download/clang+llvm-21.1.2-x86_64-pc-windows-msvc/bin/clang-cl.exe

ls -l "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm"

where cl.exe

where clang-cl.exe

export CC=/tmp/clang-download/clang+llvm-21.1.2-x86_64-pc-windows-msvc/bin/clang-cl.exe
export CXX=/tmp/clang-download/clang+llvm-21.1.2-x86_64-pc-windows-msvc/bin/clang-cl.exe
export LD=link

# The CMAKE_*_LINKER_FLAGS to disable the manifest come from research
# on fixing a build reliability issue on the build server, please
# see https://github.com/llvm/llvm-project/pull/82393 and
# https://discourse.llvm.org/t/rfc-future-of-windows-pre-commit-ci/76840/40
# for further information.
# We limit the number of parallel compile jobs to 24 control memory
# consumption and improve build reliability.
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests --succinct" \
      -D COMPILER_RT_BUILD_ORC=OFF \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_MODULE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}"

cp ${BUILD_DIR}/CMakeCache.txt ${MONOREPO_ROOT}/CMakeCache.clang1.txt

pushd ${BUILD_DIR}
rm -Rf *
popd

export CC=cl
export CXX=cl
export LD=link

cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests --succinct" \
      -D COMPILER_RT_BUILD_ORC=OFF \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_MODULE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}"

cp ${BUILD_DIR}/CMakeCache.txt ${MONOREPO_ROOT}/CMakeCache.msvc.txt

export CC=/tmp/clang-download/clang+llvm-21.1.2-x86_64-pc-windows-msvc/bin/clang-cl.exe
export CXX=/tmp/clang-download/clang+llvm-21.1.2-x86_64-pc-windows-msvc/bin/clang-cl.exe
export LD=link

cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --use-unique-output-file-name --timeout=1200 --time-tests --succinct" \
      -D COMPILER_RT_BUILD_ORC=OFF \
      -D CMAKE_C_COMPILER_LAUNCHER=sccache \
      -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_MODULE_LINKER_FLAGS="/MANIFEST:NO" \
      -D CMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO" \
      -D LLVM_ENABLE_RUNTIMES="${runtimes}"

cp ${BUILD_DIR}/CMakeCache.txt ${MONOREPO_ROOT}/CMakeCache.clang2.txt

diff ${MONOREPO_ROOT}/CMakeCache.clang1.txt ${MONOREPO_ROOT}/CMakeCache.clang2.txt


start-group "ninja"


# Targets are not escaped as they are passed as separate arguments.
ninja -C "${BUILD_DIR}" -k 0 ${targets} |& tee ninja.log

if [[ "${runtime_targets}" != "" ]]; then
  start-group "ninja runtimes"
  
  ninja -C "${BUILD_DIR}" -k 0 ${runtimes_targets} |& tee ninja_runtimes.log
fi
