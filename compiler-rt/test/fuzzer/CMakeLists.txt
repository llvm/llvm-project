set(LIBFUZZER_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if (NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND LIBFUZZER_TEST_DEPS fuzzer asan ubsan)
  if (COMPILER_RT_HAS_MSAN)
    list(APPEND LIBFUZZER_TEST_DEPS msan)
  endif()
  if (COMPILER_RT_HAS_DFSAN)
    list(APPEND LIBFUZZER_TEST_DEPS dfsan)
  endif()
  if(NOT APPLE AND COMPILER_RT_HAS_LLD)
    list(APPEND LIBFUZZER_TEST_DEPS lld)
  endif()
endif()

if (APPLE)
  darwin_filter_host_archs(FUZZER_SUPPORTED_ARCH FUZZER_SUPPORTED_ARCH)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  list(APPEND LIBFUZZER_TEST_DEPS FuzzerUnitTests)
endif()

add_custom_target(check-fuzzer)

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

macro(test_fuzzer stdlib)
  cmake_parse_arguments(TEST "" "" "DEPS" ${ARGN})
  string(REPLACE "+" "x" stdlib_name ${stdlib})
  string(REPLACE "-" ";" stdlib_list ${stdlib_name})
  set(STDLIB_CAPITALIZED "")
  foreach(part IN LISTS stdlib_list)
    string(SUBSTRING ${part} 0 1 first_letter)
    string(TOUPPER ${first_letter} first_letter)
    string(REGEX REPLACE "^.(.*)" "${first_letter}\\1" part "${part}")
    set(STDLIB_CAPITALIZED "${STDLIB_CAPITALIZED}${part}")
  endforeach()
  foreach(arch ${FUZZER_SUPPORTED_ARCH})
    set(LIBFUZZER_TEST_COMPILER ${COMPILER_RT_TEST_COMPILER})
    get_test_cc_for_arch(${arch} LIBFUZZER_TEST_COMPILER LIBFUZZER_TEST_FLAGS)

    set(LIBFUZZER_TEST_APPLE_PLATFORM "osx")

    set(LIBFUZZER_TEST_STDLIB ${stdlib})

    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${STDLIB_CAPITALIZED}${OS_NAME}Config)

    # LIT-based libFuzzer tests.
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )

    add_lit_testsuite(check-fuzzer-${stdlib_name}-${arch}
      "Running libFuzzer ${stdlib} tests for arch ${arch}"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${LIBFUZZER_TEST_DEPS})
    if(TEST_DEPS)
      add_dependencies(check-fuzzer-${stdlib_name}-${arch} ${TEST_DEPS})
    endif()
    set_target_properties(check-fuzzer-${stdlib_name}-${arch}
        PROPERTIES FOLDER "Compiler-RT Tests")
    add_dependencies(check-fuzzer check-fuzzer-${stdlib_name}-${arch})
  endforeach()
endmacro()

test_fuzzer("default")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  if(TARGET cxx_shared)
    test_fuzzer("libc++" DEPS cxx_shared)
  endif()
  if(TARGET cxx_static)
    test_fuzzer("static-libc++" DEPS cxx_static)
  endif()
endif()

if (APPLE)
  set(EXCLUDE_FROM_ALL ON)

  foreach(arch ${DARWIN_ios_ARCHS})
    set(LIBFUZZER_TEST_APPLE_PLATFORM "ios")
    set(LIBFUZZER_TEST_TARGET_ARCH ${arch})
    set(LIBFUZZER_TEST_FLAGS "-arch ${arch} -isysroot ${DARWIN_ios_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
    set(LIBFUZZER_TEST_CONFIG_SUFFIX "-${arch}-${LIBFUZZER_TEST_APPLE_PLATFORM}")
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
