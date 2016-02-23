set(LSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(LSAN_TESTSUITES)
 
set(LSAN_TEST_ARCH ${LSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(LSAN_SUPPORTED_ARCH LSAN_TEST_ARCH)
endif()

foreach(arch ${LSAN_TEST_ARCH})
  set(LSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" LSAN_TEST_CONFIG_SUFFIX)
  if(ANDROID OR ${arch} MATCHES "arm|aarch64")
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(LSAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
    set(LSAN_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
  else()
    get_target_flags_for_arch(${arch} LSAN_TEST_TARGET_CFLAGS)
    string(REPLACE ";" " " LSAN_TEST_TARGET_CFLAGS "${LSAN_TEST_TARGET_CFLAGS}")
  endif()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(LSAN_LIT_TEST_MODE "Standalone")
  set(CONFIG_NAME ${ARCH_UPPER_CASE}LsanConfig)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND LSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

  set(CONFIG_NAME ${ARCH_UPPER_CASE}AsanConfig)
  set(LSAN_LIT_TEST_MODE "AddressSanitizer")

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND LSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(LSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND LSAN_TEST_DEPS lsan asan)
endif()
add_lit_testsuite(check-lsan "Running the LeakSanitizer tests"
  ${LSAN_TESTSUITES}
  DEPENDS ${LSAN_TEST_DEPS})
set_target_properties(check-lsan PROPERTIES FOLDER "LSan tests")
