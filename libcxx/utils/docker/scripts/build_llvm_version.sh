#!/usr/bin/env bash
#===- libcxx/utils/docker/scripts/build_install_llvm_version_default.sh -----------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===-------------------------------------------------------------------------------------------===//

set -e

function show_usage() {
  cat << EOF
Usage: build_install_llvm.sh [options] -- [cmake-args]

Run cmake with the specified arguments. Used inside docker container.
Passes additional -DCMAKE_INSTALL_PREFIX and puts the build results into
the directory specified by --to option.

Available options:
  -h|--help           show this help message
  --install           destination directory where to install the targets.
  --branch            the branch or tag of LLVM to build
Required options: --install, and --version.

All options after '--' are passed to CMake invocation.
EOF
}

LLVM_BRANCH=""
CMAKE_ARGS=""
LLVM_INSTALL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      shift
      LLVM_INSTALL_DIR="$1"
      shift
      ;;
    --branch)
      shift
      LLVM_BRANCH="$1"
      shift
      ;;
    --)
      shift
      CMAKE_ARGS="$*"
      shift $#
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
  esac
done


if [ "$LLVM_INSTALL_DIR" == "" ]; then
  echo "No install directory. Please specify the --install argument."
  exit 1
fi

if [ "$LLVM_BRANCH" == "" ]; then
  echo "No install directory. Please specify the --branch argument."
  exit 1
fi

if [ "$CMAKE_ARGS" == "" ]; then
  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RELEASE '-DCMAKE_C_FLAGS=-gline-tables-only' '-DCMAKE_CXX_FLAGS=-gline-tables-only' -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON"
fi

set -x

TMP_ROOT="$(mktemp -d -p /tmp)"
LLVM_SOURCE_DIR="$TMP_ROOT/llvm-project"
LLVM_BUILD_DIR="$TMP_ROOT/build"
LLVM="$LLVM_SOURCE_DIR/llvm"

git clone --branch $LLVM_BRANCH --single-branch --depth=1 https://github.com/llvm/llvm-project.git $LLVM_SOURCE_DIR

pushd "$LLVM_SOURCE_DIR"

# Setup the source-tree using the old style layout
ln -s $LLVM_SOURCE_DIR/libcxx $LLVM/projects/libcxx
ln -s $LLVM_SOURCE_DIR/libcxxabi $LLVM/projects/libcxxabi
ln -s $LLVM_SOURCE_DIR/compiler-rt $LLVM/projects/compiler-rt
ln -s $LLVM_SOURCE_DIR/clang $LLVM/tools/clang
ln -s $LLVM_SOURCE_DIR/clang-tools-extra $LLVM/tools/clang/tools/extra

popd

# Configure and build
mkdir "$LLVM_BUILD_DIR"
pushd "$LLVM_BUILD_DIR"
cmake -GNinja "-DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR" $CMAKE_ARGS $LLVM
ninja install
popd

# Cleanup
rm -rf "$TMP_ROOT/"

echo "Done"
