configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg.py)
list(APPEND SCUDO_STANDALONE_TEST_DEPS ScudoUnitTests)
list(APPEND SCUDO_STANDALONE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/unit)
if (COMPILER_RT_HAS_GWP_ASAN)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gwp_asan/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/unit/gwp_asan/lit.site.cfg.py)
  list(APPEND SCUDO_STANDALONE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/unit/gwp_asan)
endif()

add_lit_testsuite(check-scudo_standalone
  "Running Scudo Standalone tests"
  ${SCUDO_STANDALONE_TESTSUITES}
  DEPENDS ${SCUDO_STANDALONE_TEST_DEPS})

set_target_properties(check-scudo_standalone
  PROPERTIES FOLDER "Compiler-RT Tests")
