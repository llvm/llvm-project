set(SANITIZER_COMMON_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SANITIZER_COMMON_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
set(SANITIZER_COMMON_TESTSUITES)

set(SUPPORTED_TOOLS)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux|FreeBSD|NetBSD|SunOS")
  list(APPEND SUPPORTED_TOOLS asan)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND NOT ANDROID)
  list(APPEND SUPPORTED_TOOLS tsan)
  list(APPEND SUPPORTED_TOOLS msan)
  list(APPEND SUPPORTED_TOOLS lsan)
  list(APPEND SUPPORTED_TOOLS ubsan)
endif()

# Create a separate config for each tool we support.
foreach(tool ${SUPPORTED_TOOLS})
  string(TOUPPER ${tool} tool_toupper)
  if(${tool_toupper}_SUPPORTED_ARCH AND NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND SANITIZER_COMMON_TEST_DEPS ${tool})
  endif()
  set(TEST_ARCH ${${tool_toupper}_SUPPORTED_ARCH})
  if(APPLE)
    darwin_filter_host_archs(${tool_toupper}_SUPPORTED_ARCH TEST_ARCH)
  endif()

  foreach(arch ${TEST_ARCH})
    set(SANITIZER_COMMON_LIT_TEST_MODE ${tool})
    set(SANITIZER_COMMON_TEST_TARGET_ARCH ${arch})
    get_test_cc_for_arch(${arch} SANITIZER_COMMON_TEST_TARGET_CC SANITIZER_COMMON_TEST_TARGET_CFLAGS)
    set(CONFIG_NAME ${tool}-${arch}-${OS_NAME})
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
    list(APPEND SANITIZER_COMMON_TESTSUITES
         ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endforeach()
endforeach()

# Unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  # FIXME: support unit test in the android test runner
  if (NOT ANDROID)
    list(APPEND SANITIZER_COMMON_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
    list(APPEND SANITIZER_COMMON_TEST_DEPS SanitizerUnitTests)
  endif()
endif()

if(SANITIZER_COMMON_TESTSUITES)
  add_lit_testsuite(check-sanitizer "Running sanitizer_common tests"
    ${SANITIZER_COMMON_TESTSUITES}
    DEPENDS ${SANITIZER_COMMON_TEST_DEPS})
  set_target_properties(check-sanitizer PROPERTIES FOLDER
                        "Compiler-RT Misc")
endif()
