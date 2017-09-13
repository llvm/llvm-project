set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UBSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(UBSAN_SUPPORTED_ARCH UBSAN_TEST_ARCH)
endif()

set(UBSAN_TESTSUITES)
set(UBSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND UBSAN_TEST_DEPS ubsan-minimal)
endif()

foreach(arch ${UBSAN_TEST_ARCH})
  get_test_cc_for_arch(${arch} UBSAN_TEST_TARGET_CC UBSAN_TEST_TARGET_CFLAGS)
  set(CONFIG_NAME ${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND UBSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-ubsan-minimal "Running UndefinedBehaviorSanitizerMinimal tests"
  ${UBSAN_TESTSUITES}
  DEPENDS ${UBSAN_TEST_DEPS})
set_target_properties(check-ubsan-minimal PROPERTIES FOLDER "Compiler-RT Misc")
