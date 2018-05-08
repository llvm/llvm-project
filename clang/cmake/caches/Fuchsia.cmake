# This file sets up a CMakeCache for a Fuchsia toolchain build.

set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(CLANG_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(BOOTSTRAP_LLVM_ENABLE_LTO ON CACHE BOOL "")
if(NOT APPLE)
  set(BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
endif()

if(APPLE)
  set(COMPILER_RT_ENABLE_IOS OFF CACHE BOOL "")
  set(COMPILER_RT_ENABLE_TVOS OFF CACHE BOOL "")
  set(COMPILER_RT_ENABLE_WATCHOS OFF CACHE BOOL "")
endif()

set(CLANG_BOOTSTRAP_TARGETS
  check-all
  check-llvm
  check-clang
  llvm-config
  test-suite
  test-depends
  llvm-test-depends
  clang-test-depends
  distribution
  install-distribution
  clang CACHE STRING "")

get_cmake_property(variableNames VARIABLES)
foreach(variableName ${variableNames})
  if(variableName MATCHES "^STAGE2_")
    string(REPLACE "STAGE2_" "" new_name ${variableName})
    list(APPEND EXTRA_ARGS "-D${new_name}=${${variableName}}")
  endif()
endforeach()

# Setup the bootstrap build.
set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS
  ${EXTRA_ARGS}
  -C ${CMAKE_CURRENT_LIST_DIR}/Fuchsia-stage2.cmake
  CACHE STRING "")
