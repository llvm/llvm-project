# This file sets up a CMakeCache for the second stage of a Fuchsia toolchain build.

include(${CMAKE_CURRENT_LIST_DIR}/Fuchsia-stage2.cmake)

if(NOT APPLE)
  set(BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
endif()

set(CLANG_BOOTSTRAP_TARGETS
  check-all
  check-clang
  check-lld
  check-llvm
  check-polly
  clang
  clang-test-depends
  toolchain-distribution
  install-toolchain-distribution
  install-toolchain-distribution-stripped
  install-toolchain-distribution-toolchain
  lld-test-depends
  llvm-config
  llvm-test-depends
  test-depends
  test-suite CACHE STRING "")

get_cmake_property(variableNames VARIABLES)
foreach(variableName ${variableNames})
  if(variableName MATCHES "^STAGE2_")
    string(REPLACE "STAGE2_" "" new_name ${variableName})
    list(APPEND EXTRA_ARGS "-D${new_name}=${${variableName}}")
  endif()
endforeach()

set(CLANG_PGO_TRAINING_DEPS
  builtins
  runtimes
  CACHE STRING "")

# Setup the bootstrap build.
set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS
  ${EXTRA_ARGS}
  -C ${CMAKE_CURRENT_LIST_DIR}/Fuchsia-stage2.cmake
  CACHE STRING "")
