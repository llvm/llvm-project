set(CSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Device compiler-rt only builds the GPU archive, the tests run from the host.
if(COMPILER_RT_GPU_BUILD)
  return()
endif()

set(CSAN_TESTSUITES)
set(CSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND CSAN_TEST_DEPS csan)

set(CSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
foreach(arch ${CSAN_TEST_ARCH})
  set(CSAN_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} CSAN_TEST_TARGET_CC CSAN_TEST_TARGET_CFLAGS)
  set(CONFIG_NAME ${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND CSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-csan "Running ConcurrencySanitizer tests"
  ${CSAN_TESTSUITES}
  DEPENDS ${CSAN_TEST_DEPS})
