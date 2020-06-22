# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

set(FLANG_INTRINSIC_MODULES_DIR ${FLANG_BINARY_DIR}/include/flang)

set(FLANG_TOOLS_DIR ${FLANG_BINARY_DIR}/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_TEST_DEPENDS
  f18 FileCheck count not module_files
)

if (LINK_WITH_FIR)
  list(APPEND FLANG_TEST_DEPENDS tco)
endif()

if (FLANG_INCLUDE_TESTS)
  if (FLANG_GTEST_AVAIL)
    list(APPEND FLANG_TEST_DEPENDS FlangUnitTests)
  endif()
endif()

add_custom_target(flang-test-depends DEPENDS ${FLANG_TEST_DEPENDS})

add_lit_testsuite(check-flang "Running the Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENDS}
)
set_target_properties(check-flang PROPERTIES FOLDER "Tests")

add_lit_testsuites(FLANG ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENDS})

