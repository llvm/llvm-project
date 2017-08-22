set(LIBFUZZER_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if (NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND LIBFUZZER_TEST_DEPS fuzzer asan)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  list(APPEND LIBFUZZER_TEST_DEPS FuzzerUnitTests)
endif()

set(LIBFUZZER_TESTSUITES)


if(COMPILER_RT_INCLUDE_TESTS)
  # libFuzzer unit tests.
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg)
  list(APPEND LIBFUZZER_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/unit)
endif()

foreach(arch ${FUZZER_SUPPORTED_ARCH})
  set(LIBFUZZER_TEST_COMPILER ${COMPILER_RT_TEST_COMPILER})
  get_test_cc_for_arch(${arch} LIBFUZZER_TEST_COMPILER LIBFUZZER_TEST_FLAGS)

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  # LIT-based libFuzzer tests.
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
    )
  list(APPEND LIBFUZZER_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

endforeach()

set(EXCLUDE_FROM_ALL ON)

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
  ${LIBFUZZER_TESTSUITES}
  DEPENDS ${LIBFUZZER_TEST_DEPS})
set_target_properties(check-fuzzer PROPERTIES FOLDER "Compiler-RT Tests")
