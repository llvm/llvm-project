set(BITINT_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BITINT_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} bitint)
set(BITINT_TESTSUITES)

include(builtin-config-ix)

foreach(arch ${BITINT_SUPPORTED_ARCH})
  set(BITINT_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" BITINT_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} BITINT_TEST_TARGET_CC BITINT_TEST_TARGET_CFLAGS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${arch}/lit.site.cfg.py
    )
  list(APPEND BITINT_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${arch})
endforeach()

add_lit_testsuite(check-bitint "Running the Bitint tests"
  ${BITINT_TESTSUITES}
  DEPENDS ${BITINT_TEST_DEPS})
set_target_properties(check-bitint PROPERTIES FOLDER "Compiler-RT Misc")

