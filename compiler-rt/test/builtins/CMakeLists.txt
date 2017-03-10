set(BUILTINS_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BUILTINS_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} builtins)
set(BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/TestCases)

# Test cases.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

add_lit_testsuite(check-builtins "Running the Builtins tests"
  ${BUILTINS_TESTSUITES}
  DEPENDS ${BUILTINS_TEST_DEPS})
set_target_properties(check-builtins PROPERTIES FOLDER "Compiler-RT Misc")
