set(CMAKE_ASM_COMPILER_TARGET "wasm32-wasip1" CACHE STRING "")
set(CMAKE_CXX_COMPILER_TARGET "wasm32-wasip1" CACHE STRING "")
set(CMAKE_C_COMPILER_TARGET "wasm32-wasip1" CACHE STRING "")
set(CMAKE_SYSTEM_NAME Generic CACHE STRING "")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY CACHE STRING "")

set(COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_PROFILE OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_SANITIZERS OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_XRAY OFF CACHE BOOL "")
set(COMPILER_RT_DEFAULT_TARGET_ONLY ON CACHE BOOL "")

set(LIBCXX_ENABLE_EXCEPTIONS ON CACHE BOOL "")
set(LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
# The locale base API has no WASI backend, so it falls back to the IBM one.
set(LIBCXX_ENABLE_LOCALIZATION OFF CACHE BOOL "")
set(LIBCXX_ENABLE_MONOTONIC_CLOCK ON CACHE BOOL "")
set(LIBCXX_ENABLE_RTTI ON CACHE BOOL "")
set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXX_ENABLE_STATIC ON CACHE BOOL "")
set(LIBCXX_ENABLE_THREADS OFF CACHE BOOL "")
set(LIBCXX_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
# Long tests are prohibitively slow when run under a WebAssembly interpreter.
set(LIBCXX_TEST_PARAMS "long_tests=False" CACHE STRING "")
set(LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")

set(LIBCXXABI_BAREMETAL ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_EXCEPTIONS ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXXABI_ENABLE_STATIC ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_THREADS OFF CACHE BOOL "")
set(LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
set(LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")

set(LIBUNWIND_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBUNWIND_ENABLE_STATIC ON CACHE BOOL "")
set(LIBUNWIND_ENABLE_THREADS OFF CACHE BOOL "")
set(LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")

find_program(WASI_RUNTIME wasmkit REQUIRED)

# On embedded platforms that don't support shared library targets, CMake implicitly changes shared
# library targets to be static library targets. This results in duplicate definitions of the static
# library targets even though we might not ever build the shared library target, which breaks the
# build. To work around this, we change the output name of the  shared library target so that it
# can't conflict with the static library target.
#
# This is tracked by https://gitlab.kitware.com/cmake/cmake/-/issues/25759.
set(LIBCXX_SHARED_OUTPUT_NAME "c++-shared" CACHE STRING "")
set(LIBCXXABI_SHARED_OUTPUT_NAME "c++abi-shared" CACHE STRING "")
set(LIBUNWIND_SHARED_OUTPUT_NAME "unwind-shared" CACHE STRING "")
