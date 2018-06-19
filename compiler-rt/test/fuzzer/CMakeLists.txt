set(LIBFUZZER_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(REMOVE_ITEM LIBFUZZER_TEST_DEPS SanitizerLintCheck)
if (NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND LIBFUZZER_TEST_DEPS fuzzer asan ubsan)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  list(APPEND LIBFUZZER_TEST_DEPS FuzzerUnitTests)
endif()

add_custom_target(check-fuzzer)

set(LIBFUZZER_TEST_IOS "0")
pythonize_bool(LIBFUZZER_TEST_IOS)

if(COMPILER_RT_INCLUDE_TESTS)
  # libFuzzer unit tests.
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg)
  add_lit_testsuite(check-fuzzer-unit "Running Fuzzer unit tests"
    ${CMAKE_CURRENT_BINARY_DIR}/unit
    DEPENDS ${LIBFUZZER_TEST_DEPS})
  set_target_properties(check-fuzzer-unit PROPERTIES FOLDER "Compiler-RT Tests")
  add_dependencies(check-fuzzer check-fuzzer-unit)
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
  add_lit_testsuite(check-fuzzer-${arch}
    "Running Fuzzer ${arch} tests"
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
    DEPENDS ${LIBFUZZER_TEST_DEPS})
  if (TEST_DEPS)
    add_dependencies(check-fuzzer-${arch} ${TEST_DEPS})
  endif()
  set_target_properties(check-fuzzer-${arch} PROPERTIES FOLDER "Compiler-RT Tests")
  add_dependencies(check-fuzzer check-fuzzer-${arch})
endforeach()

set(EXCLUDE_FROM_ALL ON)

if (APPLE)
  set(EXCLUDE_FROM_ALL ON)

  foreach(arch ${DARWIN_ios_ARCHS})
    set(LIBFUZZER_TEST_IOS "1")
    pythonize_bool(LIBFUZZER_TEST_IOS)
    set(LIBFUZZER_TEST_TARGET_ARCH ${arch})
    set(LIBFUZZER_TEST_FLAGS "-arch ${arch} -isysroot ${DARWIN_ios_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
    set(LIBFUZZER_TEST_CONFIG_SUFFIX "-${arch}-ios")
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME "IOS${ARCH_UPPER_CASE}Config")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )
    add_lit_testsuite(check-fuzzer-ios-${arch} "libFuzzer iOS ${arch} tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${LIBFUZZER_TEST_DEPS})

  endforeach()

  set(EXCLUDE_FROM_ALL OFF)
endif()
