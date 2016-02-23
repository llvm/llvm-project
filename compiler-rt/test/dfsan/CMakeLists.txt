set(DFSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(DFSAN_TESTSUITES)
 
set(DFSAN_TEST_ARCH ${DFSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(DFSAN_SUPPORTED_ARCH DFSAN_TEST_ARCH)
endif()

foreach(arch ${DFSAN_TEST_ARCH})
  set(DFSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" DFSAN_TEST_CONFIG_SUFFIX)
  if(ANDROID OR ${arch} MATCHES "arm|aarch64")
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(DFSAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
    set(DFSAN_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
  else()
    get_target_flags_for_arch(${arch} DFSAN_TEST_TARGET_CFLAGS)
    string(REPLACE ";" " " DFSAN_TEST_TARGET_CFLAGS "${DFSAN_TEST_TARGET_CFLAGS}")
  endif()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND DFSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(DFSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND DFSAN_TEST_DEPS dfsan)
endif()

add_lit_testsuite(check-dfsan "Running the DataFlowSanitizer tests"
  ${DFSAN_TESTSUITES}
  DEPENDS ${DFSAN_TEST_DEPS})
set_target_properties(check-dfsan PROPERTIES FOLDER "DFSan tests")
