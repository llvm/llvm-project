set(RADSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(RADSAN_TESTSUITES)
set(RADSAN_FDR_TESTSUITES)

set(RADSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(RADSAN_FDR_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(RADSAN_TEST_ARCH ${RADSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(RADSAN_SUPPORTED_ARCH RADSAN_TEST_ARCH)
endif()

if (COMPILER_RT_HAS_RADSAN)
  foreach(arch ${RADSAN_TEST_ARCH})
    set(RADSAN_TEST_TARGET_ARCH ${arch})
    string(TOLOWER "-${arch}-${OS_NAME}" RADSAN_TEST_CONFIG_SUFFIX)
    get_test_cc_for_arch(${arch} RADSAN_TEST_TARGET_CC RADSAN_TEST_TARGET_CFLAGS)
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND RADSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endforeach()
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
  if(COMPILER_RT_RADSAN_HAS_STATIC_RUNTIME)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic/lit.site.cfg.py)
  endif()
  list(APPEND RADSAN_TEST_DEPS RadsanUnitTests)
  list(APPEND RADSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
  if(COMPILER_RT_RADSAN_HAS_STATIC_RUNTIME)
    list(APPEND RADSAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic)
  endif()
endif()

add_lit_testsuite(check-radsan "Running the Radsan tests"
  ${RADSAN_TESTSUITES}
  DEPENDS ${RADSAN_TEST_DEPS})
set_target_properties(check-radsan PROPERTIES FOLDER "Compiler-RT Misc")
