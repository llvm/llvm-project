cmake_minimum_required(VERSION 3.18)
project(LoadStoreAnalysisPass)

find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LoadStoreAnalysisPass MODULE LoadStoreAnalysisPass.cpp)

