set(BUILTINS_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BUILTINS_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} builtins)
set(BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/TestCases)

# Test cases.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

#Unit tests.

include(builtin-config-ix)

foreach(arch ${BUILTIN_SUPPORTED_ARCH})
  set(BUILTINS_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" BUILTINS_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} BUILTINS_TEST_TARGET_CC BUILTINS_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg
    )
  list(APPEND BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-builtins "Running the Builtins tests"
  ${BUILTINS_TESTSUITES}
  DEPENDS ${BUILTINS_TEST_DEPS})
set_target_properties(check-builtins PROPERTIES FOLDER "Compiler-RT Misc")
