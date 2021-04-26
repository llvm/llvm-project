set(ORC_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ORC_TESTSUITES)
set(ORC_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(ORC_TEST_ARCH ${ORC_SUPPORTED_ARCH})
if (COMPILER_RT_BUILD_ORC AND COMPILER_RT_HAS_ORC)
  foreach(arch ${ORC_TEST_ARCH})
    set(ORC_TEST_TARGET_ARCH ${arch})
    string(TOLOWER "-${arch}-${OS_NAME}" ORC_TEST_CONFIG_SUFFIX)
    get_test_cc_for_arch(${arch} ORC_TEST_TARGET_CC ORC_TEST_TARGET_CFLAGS)
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND ORC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endforeach()
endif()

add_lit_testsuite(check-orc "Running the ORC runtime tests"
  ${ORC_TESTSUITES}
  DEPENDS ${ORC_TEST_DEPS})
set_target_properties(check-orc PROPERTIES FOLDER "Compiler-RT Misc")
