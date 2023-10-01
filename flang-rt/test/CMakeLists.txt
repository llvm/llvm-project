# Test runner infrastructure for Flang-rt. This configures the Flang-rt test
# trees for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  FLANG_STANDALONE_BUILD
  LLVM_BUILD_EXAMPLES
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_ENABLE_PLUGINS
)

set(FLANG_TOOLS_DIR ${FLANG_BINARY_DIR}/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/NonGtestUnit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/NonGtestUnit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/NonGtestUnit/lit.cfg.py
)

set(FLANG_RT_TEST_PARAMS
  flang_rt_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_RT_TEST_DEPENDS
  LLVMSupport
  flang-rt
  Fortran_main
)
if (LLVM_ENABLE_PLUGINS AND NOT WIN32)
  list(APPEND FLANG_RT_TEST_DEPENDS Bye)
endif()

if (FLANG_RT_INCLUDE_TESTS)
  if (FLANG_RT_GTEST_AVAIL)
    list(APPEND FLANG_RT_TEST_DEPENDS FlangRTUnitTests)
  endif()
endif()

add_custom_target(flang-rt-test-depends DEPENDS ${FLANG_RT_TEST_DEPENDS})

add_lit_testsuite(check-flang-rt "Running the Flang-rt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_RT_TEST_PARAMS}
  DEPENDS ${FLANG_RT_TEST_DEPENDS}
)
set_target_properties(check-flang-rt PROPERTIES FOLDER "Tests")

add_lit_testsuites(FLANG_RT ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${FLANG_RT_TEST_PARAMS}
  DEPENDS ${FLANG_RT_TEST_DEPENDS})

# To modify the default target triple for flang-rt tests.
if (DEFINED FLANG_RT_TEST_TARGET_TRIPLE)
  if (NOT DEFINED LLVM_TARGET_TRIPLE_ENV OR LLVM_TARGET_TRIPLE_ENV STREQUAL "")
    message(FATAL_ERROR "LLVM_TARGET_TRIPLE_ENV must also be defined in order "
                        "to use FLANG_RT_TEST_TARGET_TRIPLE.")
  endif()
endif()
