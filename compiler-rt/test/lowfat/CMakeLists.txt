set(LOWFAT_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(LOWFAT_TESTSUITES)
set(LOWFAT_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND LOWFAT_TEST_DEPS lowfat)

set(LOWFAT_TEST_ARCH ${LOWFAT_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(LOWFAT_SUPPORTED_ARCH LOWFAT_TEST_ARCH)
endif()

foreach(arch ${LOWFAT_TEST_ARCH})
  set(LOWFAT_TEST_TARGET_ARCH ${arch})
  # Use COMPILER_RT_TEST_COMPILER directly — get_test_cc_for_arch only populates
  # the CC output when cross-compiling; on Darwin native builds it returns empty.
  set(LOWFAT_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  get_test_cc_for_arch(${arch} _unused LOWFAT_TEST_TARGET_CFLAGS)
  string(TOLOWER "-${arch}-${OS_NAME}" LOWFAT_TEST_CONFIG_SUFFIX)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    MAIN_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )
  list(APPEND LOWFAT_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-lowfat "Running LowFat Sanitizer tests"
  ${LOWFAT_TESTSUITES}
  DEPENDS ${LOWFAT_TEST_DEPS}
)
set_target_properties(check-lowfat PROPERTIES FOLDER "Compiler-RT/Tests")
