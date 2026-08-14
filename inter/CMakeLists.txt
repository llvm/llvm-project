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

option(INTER_INCLUDE_INTEGRATION_TESTS
       "Build and enable Intel Xe hardware integration tests" OFF)
set(INTER_INTEGRATION_DEVICE_NAME "B60" CACHE STRING
    "Substring required in the Level Zero integration-test device name")

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include(cmake/FetchGED.cmake)

include_directories(BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

set(INTER_PIPELINES_FILE
    "${PROJECT_BINARY_DIR}/share/inter/pipelines/pipelines.mlir")
get_filename_component(INTER_PIPELINES_DIR "${INTER_PIPELINES_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${INTER_PIPELINES_DIR}")
configure_file(lib/inter/pipelines/pipelines.mlir "${INTER_PIPELINES_FILE}"
               COPYONLY)
install(FILES lib/inter/pipelines/pipelines.mlir
        DESTINATION share/inter/pipelines)

add_subdirectory(include/inter/Dialect/XeMachine/IR)
add_subdirectory(include/inter/Dialect/Inter/IR)
add_subdirectory(include/inter/Transforms)
add_subdirectory(lib/inter/Dialect/XeMachine/IR)
add_subdirectory(lib/inter/Dialect/Inter/IR)
add_subdirectory(lib/inter/Analysis)
add_subdirectory(lib/inter/Transforms)
add_subdirectory(lib/inter/Emit)
add_subdirectory(lib/inter/Compiler)
add_subdirectory(tools/inter-opt)
add_subdirectory(tools/inter-translate)
add_subdirectory(test)
if(INTER_INCLUDE_INTEGRATION_TESTS)
  add_subdirectory(benchmarks)
endif()
