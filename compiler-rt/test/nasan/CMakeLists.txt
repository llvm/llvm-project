set(NASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(NASAN_TESTSUITES)
set(NASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} nasan)

macro(add_nasan_testsuite arch)
  set(NASAN_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} NASAN_TEST_TARGET_CC NASAN_TEST_TARGET_CFLAGS)

  string(TOUPPER ${arch} CONFIG_NAME)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND NASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endmacro()

set(NASAN_TEST_ARCH ${NASAN_SUPPORTED_ARCH})
foreach(arch ${NASAN_TEST_ARCH})
  add_nasan_testsuite(${arch})
endforeach()

add_lit_testsuite(check-nasan "Running the NoAlias sanitizer tests"
  ${NASAN_TESTSUITES}
  DEPENDS ${NASAN_TEST_DEPS}
  )
