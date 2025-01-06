set(RTSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(RTSAN_TESTSUITES)
set(RTSAN_FDR_TESTSUITES)

set(RTSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(RTSAN_FDR_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(RTSAN_TEST_ARCH ${RTSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(RTSAN_SUPPORTED_ARCH RTSAN_TEST_ARCH)
endif()

foreach(arch ${RTSAN_TEST_ARCH})
  set(RTSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" RTSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} RTSAN_TEST_TARGET_CC RTSAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND RTSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
if(COMPILER_RT_RTSAN_HAS_STATIC_RUNTIME)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic/lit.site.cfg.py)
endif()
list(APPEND RTSAN_TEST_DEPS RtsanUnitTests)
list(APPEND RTSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
if(COMPILER_RT_RTSAN_HAS_STATIC_RUNTIME)
  list(APPEND RTSAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic)
endif()

add_lit_testsuite(check-rtsan "Running the Rtsan tests"
  ${RTSAN_TESTSUITES}
  DEPENDS ${RTSAN_TEST_DEPS})
set_target_properties(check-rtsan PROPERTIES FOLDER "Compiler-RT Misc")
