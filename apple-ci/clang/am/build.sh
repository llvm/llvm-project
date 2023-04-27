#!/usr/bin/env bash
set -eu

SRC_DIR=$PWD/llvm-project
BUILD_DIR=$PWD/build

for arg; do
  case $arg in
    --src=*) SRC_DIR="${arg##*=}"; shift ;;
    --build=*) BUILD_DIR="${arg##*=}"; shift ;;
    *) echo "Incorrect usage." >&2; exit 1 ;;
  esac
done

echo
echo "SRC_DIR . . . . = $SRC_DIR"
echo "BUILD_DIR . . . = $BUILD_DIR"
echo

NINJA=$(xcrun -f ninja)

HOST_COMPILER_PATH=$(dirname $(xcrun -f clang))

mkdir -p $BUILD_DIR && cd $_
set -x
xcrun cmake -G Ninja \
 -DCMAKE_MAKE_PROGRAM=$NINJA \
 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_C_COMPILER=$HOST_COMPILER_PATH/clang \
 -DCMAKE_CXX_COMPILER=$HOST_COMPILER_PATH/clang++ \
 -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64" \
 -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;lldb" \
 -DLLDB_ENABLE_SWIFT_SUPPORT=OFF \
 -DLLDB_INCLUDE_TESTS=OFF \
 $SRC_DIR/llvm && $NINJA
