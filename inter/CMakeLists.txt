# Standalone MLIR subproject for the Intel Xe backend ("inter").
# Build against this repo's LLVM/MLIR build tree:
#   cmake -S inter -B inter/build -G Ninja \
#     -DMLIR_DIR=$PWD/build-m0/lib/cmake/mlir \
#     -DLLVM_DIR=$PWD/build-m0/lib/cmake/llvm
#   ninja -C inter/build
cmake_minimum_required(VERSION 3.20.0)
project(inter LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_subdirectory(include/inter/Dialect/XeMachine/IR)
add_subdirectory(lib/inter/Dialect/XeMachine/IR)
add_subdirectory(tools/inter-opt)
add_subdirectory(test)
