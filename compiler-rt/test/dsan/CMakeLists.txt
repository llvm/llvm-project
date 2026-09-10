set(DSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(DSAN_TESTSUITES)

set(DSAN_TEST_ARCH ${DSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(DSAN_SUPPORTED_ARCH DSAN_TEST_ARCH)
endif()

foreach(arch ${DSAN_TEST_ARCH})
  set(DSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" DSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} DSAN_TEST_TARGET_CC DSAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)

  set(DSAN_LIT_TEST_MODE "Standalone")
  set(CONFIG_NAME ${ARCH_UPPER_CASE}DsanConfig)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND DSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(DSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND DSAN_TEST_DEPS dsan)
add_lit_testsuite(check-dsan "Running the DoubleFreeSanitizer tests"
  ${DSAN_TESTSUITES}
  DEPENDS ${DSAN_TEST_DEPS})
