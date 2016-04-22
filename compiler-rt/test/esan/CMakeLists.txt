set(ESAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND ESAN_TEST_DEPS esan)
endif()

set(ESAN_TESTSUITES)

set(ESAN_TEST_ARCH ${ESAN_SUPPORTED_ARCH})

set(ESAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach(arch ${ESAN_TEST_ARCH})
  set(ESAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" ESAN_TEST_CONFIG_SUFFIX)
  get_target_flags_for_arch(${arch} ESAN_TEST_TARGET_CFLAGS)
  string(REPLACE ";" " " ESAN_TEST_TARGET_CFLAGS "${ESAN_TEST_TARGET_CFLAGS}")

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND ESAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

# TODO(bruening): add Unit/ tests as well

add_lit_testsuite(check-esan "Running EfficiencySanitizer tests"
  ${ESAN_TESTSUITES}
  DEPENDS ${ESAN_TEST_DEPS})
set_target_properties(check-esan PROPERTIES FOLDER "Esan tests")
