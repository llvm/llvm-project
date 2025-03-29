set(INTERCEPTION_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(INTERCEPTION_TESTSUITES)

# Unit tests. There are currently no unit tests capable to running on Apple or
# Android targets.
if(NOT ANDROID AND NOT APPLE)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
  list(APPEND INTERCEPTION_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
  list(APPEND INTERCEPTION_TEST_DEPS InterceptionUnitTests)
endif()

add_lit_testsuite(check-interception "Running the Interception tests"
  ${INTERCEPTION_TESTSUITES}
  DEPENDS ${INTERCEPTION_TEST_DEPS})
