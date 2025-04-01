set(PFP_TESTSUITES)
set(PFP_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  lld
  clang)

set(PFP_TEST_ARCH ${X86_64} ${ARM64})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)

list(APPEND PFP_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

add_lit_testsuite(check-pfp "Running the PointerFieldProtection tests"
  ${PFP_TESTSUITES}
  DEPENDS ${PFP_TEST_DEPS})

