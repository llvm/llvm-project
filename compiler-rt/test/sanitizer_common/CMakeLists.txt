set(SANITIZER_COMMON_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SANITIZER_COMMON_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

if(CMAKE_HOST_UNIX)
  list(APPEND SANITIZER_COMMON_TEST_DEPS SanitizerLintCheck)
endif()

set(SANITIZER_COMMON_TESTSUITES)

# FIXME(dliew): We should switch to COMPILER_RT_SANITIZERS_TO_BUILD instead of
# the hard coded `SUPPORTED_TOOLS_INIT` list once we know that the other
# sanitizers work.
set(SUPPORTED_TOOLS_INIT asan lsan msan tsan ubsan)
set(SUPPORTED_TOOLS)
  foreach(SANITIZER_TOOL ${SUPPORTED_TOOLS_INIT})
    string(TOUPPER ${SANITIZER_TOOL} SANITIZER_TOOL_UPPER)
    if (COMPILER_RT_HAS_${SANITIZER_TOOL_UPPER})
      list(APPEND SUPPORTED_TOOLS ${SANITIZER_TOOL})
    endif()
  endforeach()

# FIXME(dliew): Remove this.
# Temporary helper for https://reviews.llvm.org/D55740
message(
  STATUS
  "Generated Sanitizer SUPPORTED_TOOLS list on \"${CMAKE_SYSTEM_NAME}\" is"
  " \"${SUPPORTED_TOOLS}\"")

# FIXME(dliew): These tests should be made to work on all platforms.
# Use the legacy list for now.
if (ANDROID OR WINDOWS)
  set(OLD_SUPPORTED_TOOLS ${SUPPORTED_TOOLS})
  if (ANDROID)
    set(SUPPORTED_TOOLS asan)
  elseif (WINDOWS)
    set(SUPPORTED_TOOLS "")
  else()
    message(FATAL_ERROR "Unhandled platform")
  endif()
	message(
		AUTHOR_WARNING
    "Replacing Sanitizer SUPPORTED_TOOLS list (${OLD_SUPPORTED_TOOLS}) with "
    "\"${SUPPORTED_TOOLS}\"")
  unset(OLD_SUPPORTED_TOOLS)
endif()

# FIXME(dliew): Remove this.
# Temporary helper for https://reviews.llvm.org/D55740
message(
  STATUS
  "sanitizer_common tests on \"${CMAKE_SYSTEM_NAME}\" will run against "
  "\"${SUPPORTED_TOOLS}\"")

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
  if(${tool} STREQUAL "asan")
    list(REMOVE_ITEM TEST_ARCH sparc sparcv9)
  endif()
  if(OS_NAME MATCHES "SunOS" AND ${tool} STREQUAL "asan")
    list(REMOVE_ITEM TEST_ARCH x86_64)
  endif()

  # TODO(dliew): We should iterate over the different
  # Apple platforms, not just macOS.
  foreach(arch ${TEST_ARCH})
    set(SANITIZER_COMMON_LIT_TEST_MODE ${tool})
    set(SANITIZER_COMMON_TEST_TARGET_ARCH ${arch})
    get_test_cc_for_arch(${arch} SANITIZER_COMMON_TEST_TARGET_CC SANITIZER_COMMON_TEST_TARGET_CFLAGS)
    set(CONFIG_NAME ${tool}-${arch}-${OS_NAME})

    # ARM on Linux might use the slow unwinder as default and the unwind table is
    # required to get a complete stacktrace.
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" AND NOT ANDROID)
      list(APPEND SANITIZER_COMMON_TEST_TARGET_CFLAGS -funwind-tables)
      string(REPLACE ";" " " SANITIZER_COMMON_TEST_TARGET_CFLAGS
                             "${SANITIZER_COMMON_TEST_TARGET_CFLAGS}")
    endif()

    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    # FIXME(dliew): LSan i386 on Darwin is completly broken right now.
    # so don't run the tests by default.
    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin" AND
             ${tool} STREQUAL "lsan" AND
             ${arch} STREQUAL "i386"))
      list(APPEND SANITIZER_COMMON_TESTSUITES
           ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
    endif()
  endforeach()
endforeach()

# Unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
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
