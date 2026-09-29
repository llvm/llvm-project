# Second stage instrumentation (used by PGO.cmake)

set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS
  distribution
  install-distribution
  install-distribution-toolchain
  check-all
  check-llvm
  check-clang
  check-flang
  test-suite CACHE STRING "")
set(FLANG_PGO_TRAINING_CLANG_COUPLING ON CACHE BOOL "")
set(PGO_OPT_PROFDATA "${CMAKE_BINARY_DIR}/flang.profdata" CACHE STRING "")
set(PGO_OPT_PROFDATA_PROVIDER generate-flang-profdata CACHE STRING "")

if(PGO_BUILD_CONFIGURATION)
  include(${PGO_BUILD_CONFIGURATION})
  set(CLANG_BOOTSTRAP_CMAKE_ARGS
    -C ${PGO_BUILD_CONFIGURATION}
    CACHE STRING "")
else()
  include(${CMAKE_CURRENT_LIST_DIR}/PGO-stage2.cmake)

  set(CLANG_BOOTSTRAP_CMAKE_ARGS
    -C ${CMAKE_CURRENT_LIST_DIR}/PGO-stage2.cmake
    CACHE STRING "")
endif()
