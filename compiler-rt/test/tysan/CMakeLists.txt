set(TYSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TYSAN_TESTSUITES)

set(TYSAN_TEST_ARCH ${TYSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(TYSAN_SUPPORTED_ARCH TYSAN_TEST_ARCH)
endif()

foreach(arch ${TYSAN_TEST_ARCH})
  set(TYSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" TYSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} TYSAN_TEST_TARGET_CC TYSAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND TYSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(TYSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND TYSAN_TEST_DEPS tysan)
endif()

add_lit_testsuite(check-tysan "Running the TypeSanitizer tests"
  ${TYSAN_TESTSUITES}
  DEPENDS ${TYSAN_TEST_DEPS}
  )
set_target_properties(check-tysan PROPERTIES FOLDER "Compiler-RT Misc")
