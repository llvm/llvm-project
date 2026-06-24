set(COPYPROF_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(COPYPROF_TESTSUITES)

# Add lit tests.
foreach(arch ${COPYPROF_SUPPORTED_ARCH})
  set(COPYPROF_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" COPYPROF_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} COPYPROF_TEST_TARGET_CC COPYPROF_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
  )
  list(APPEND COPYPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(COPYPROF_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND COPYPROF_TEST_DEPS copyprof)

# Add unit tests.
foreach(arch ${COPYPROF_SUPPORTED_ARCH})
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND COPYPROF_TEST_DEPS CopyProfUnitTests)
  list(APPEND COPYPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-copyprof "Running the CopyProf tests"
  ${COPYPROF_TESTSUITES}
  DEPENDS ${COPYPROF_TEST_DEPS})
