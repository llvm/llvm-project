set(INTERCEPTION_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(INTERCEPTION_TESTSUITES)
set(INTERCEPTION_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach(arch ${SANITIZER_COMMON_SUPPORTED_ARCH})
  set(INTERCEPTION_TEST_TARGET_ARCH ${arch})
  set(CONFIG_NAME ${arch}-${OS_NAME})

  get_test_cc_for_arch(${arch} INTERCEPTION_TEST_TARGET_CC
    INTERCEPTION_TEST_TARGET_CFLAGS)

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  set(INTERCEPTION_TEST_TARGET_LIB RTInterception.test.${arch})

  get_property(INTERCEPTION_TEST_TARGET_LIB_DIR TARGET
    ${INTERCEPTION_TEST_TARGET_LIB} PROPERTY ARCHIVE_OUTPUT_DIRECTORY)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
    )

  list(APPEND INTERCEPTION_TESTSUITES
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  list(APPEND INTERCEPTION_TEST_DEPS RTInterception.test.${arch})
endforeach()

# Unit tests. There are currently no unit tests capable to running on Apple or
# Android targets.
if(COMPILER_RT_INCLUDE_TESTS AND NOT ANDROID AND NOT APPLE)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND INTERCEPTION_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
  list(APPEND INTERCEPTION_TEST_DEPS InterceptionUnitTests)
endif()

add_lit_testsuite(check-interception "Running the Interception tests"
  ${INTERCEPTION_TESTSUITES}
  DEPENDS ${INTERCEPTION_TEST_DEPS})
set_target_properties(check-interception PROPERTIES FOLDER "Compiler-RT Misc")
