# Two-stage build of Flang with the 2nd stage optimized using BOLT and PGO

set(BOLT_PGO_CMAKE_CACHE "PGO" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "bolt;clang;flang;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;flang-rt;libunwind;openmp" CACHE STRING "")

set(CLANG_BOOTSTRAP_TARGETS
  stage2-clang-bolt
  stage2-flang-bolt
  stage2-check-clang
  stage2-check-flang
  stage2-distribution
  stage2-install-distribution
  CACHE STRING "")
set(BOOTSTRAP_CLANG_BOOTSTRAP_TARGETS
  clang-bolt
  flang-bolt
  check-clang
  check-flang
  distribution
  install-distribution
  CACHE STRING "")

set(PGO_BUILD_CONFIGURATION
  ${CMAKE_CURRENT_LIST_DIR}/BOLT.cmake
  CACHE STRING "")
include(${CMAKE_CURRENT_LIST_DIR}/${BOLT_PGO_CMAKE_CACHE}.cmake)
