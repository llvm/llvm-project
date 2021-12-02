# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  FLANG_BUILD_EXAMPLES
  FLANG_STANDALONE_BUILD
  LLVM_ENABLE_PLUGINS
)

set(FLANG_TOOLS_DIR ${FLANG_BINARY_DIR}/bin)

# FIXME In out-of-tree builds, "SHLIBDIR" is undefined and passing it to
# `configure_lit_site_cfg` leads to a configuration error. This is currently
# only required by plugins/examples, which are not supported in out-of-tree
# builds.
if (FLANG_STANDALONE_BUILD)
  set(PATHS_FOR_PLUGINS "")
else ()
  set(PATHS_FOR_PLUGINS "SHLIBDIR")
endif ()

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

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_TEST_DEPENDS
  flang-new llvm-config FileCheck count not module_files fir-opt tco
)

if (FLANG_INCLUDE_TESTS)
  if (FLANG_GTEST_AVAIL)
    list(APPEND FLANG_TEST_DEPENDS FlangUnitTests)
  endif()
endif()

if (FLANG_BUILD_EXAMPLES)
  list(APPEND FLANG_TEST_DEPENDS
    flangPrintFunctionNames
    flangOmpReport
    )
endif ()

add_custom_target(flang-test-depends DEPENDS ${FLANG_TEST_DEPENDS})

add_lit_testsuite(check-flang "Running the Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENDS}
)
set_target_properties(check-flang PROPERTIES FOLDER "Tests")

# In case of standalone builds.
if (FLANG_STANDALONE_BUILD)
  add_lit_testsuites(FLANG ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${FLANG_TEST_PARAMS}
    DEPENDS ${FLANG_TEST_DEPENDS})
else()
  add_lit_testsuites(FLANG ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS ${FLANG_TEST_PARAMS}
    DEPENDS ${FLANG_TEST_DEPENDS})
endif()
