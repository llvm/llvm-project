cmake_minimum_required(VERSION 3.20)
project(TimeBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the libc source directory
set(LIBC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBC_SRC_DIR}
    ${LIBC_SRC_DIR}/../include
    ${LIBC_SRC_DIR}/../hdr
)

# Define LIBC_NAMESPACE macros to match the project
add_compile_definitions(
    LIBC_NAMESPACE=__llvm_libc
    LIBC_NAMESPACE_DECL=__llvm_libc
)

# Build the benchmark executable
add_executable(benchmark_time_conversion
    benchmark_time_conversion.cpp
    time_utils.cpp
    fast_date.cpp
)

# Ensure it's built with optimizations
target_compile_options(benchmark_time_conversion PRIVATE -O3 -march=native -fno-omit-frame-pointer)

# Optional: Create a simple build script
add_custom_target(run_benchmark
    COMMAND benchmark_time_conversion
    DEPENDS benchmark_time_conversion
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running time conversion benchmark"
)

message(STATUS "Benchmark configured")
message(STATUS "Build with: cmake --build . --target benchmark_time_conversion")
message(STATUS "Run with: ./benchmark_time_conversion")
