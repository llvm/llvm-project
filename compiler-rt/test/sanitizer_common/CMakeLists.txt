set(SANITIZER_COMMON_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SANITIZER_COMMON_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(SANITIZER_COMMON_TESTSUITES)

# FIXME(dliew): We should switch to COMPILER_RT_SANITIZERS_TO_BUILD instead of
# the hard coded `SUPPORTED_TOOLS_INIT` list once we know that the other
# sanitizers work.
set(SUPPORTED_TOOLS_INIT asan lsan hwasan msan tsan ubsan)
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
  if(${tool_toupper}_SUPPORTED_ARCH AND (NOT COMPILER_RT_STANDALONE_BUILD OR LLVM_RUNTIMES_BUILD))
    list(APPEND SANITIZER_COMMON_TEST_DEPS ${tool})
  endif()
  set(TEST_ARCH ${${tool_toupper}_SUPPORTED_ARCH})
  if(APPLE)
    darwin_filter_host_archs(${tool_toupper}_SUPPORTED_ARCH TEST_ARCH)
  endif()
  if(${tool} STREQUAL "asan")
    list(REMOVE_ITEM TEST_ARCH sparcv9)
  endif()
  if(OS_NAME MATCHES "SunOS" AND ${tool} STREQUAL "asan")
    list(REMOVE_ITEM TEST_ARCH x86_64)
  endif()

  foreach(arch ${TEST_ARCH})
    set(SANITIZER_COMMON_LIT_TEST_MODE ${tool})
    set(SANITIZER_COMMON_TEST_TARGET_ARCH ${arch})
    get_test_cc_for_arch(${arch} SANITIZER_COMMON_TEST_TARGET_CC SANITIZER_COMMON_TEST_TARGET_CFLAGS)
    set(CONFIG_NAME ${tool}-${arch}-${OS_NAME})

    # ARM on Linux might use the slow unwinder as default and the unwind table is
    # required to get a complete stacktrace.
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" AND NOT ANDROID)
      list(APPEND SANITIZER_COMMON_TEST_TARGET_CFLAGS -funwind-tables)
      if(CMAKE_SYSROOT)
        list(APPEND SANITIZER_COMMON_TEST_TARGET_CFLAGS "--sysroot=${CMAKE_SYSROOT}")
      endif()
      string(REPLACE ";" " " SANITIZER_COMMON_TEST_TARGET_CFLAGS
                             "${SANITIZER_COMMON_TEST_TARGET_CFLAGS}")
    endif()

    set(SANITIZER_COMMON_TEST_APPLE_TARGET_IS_HOST ON)
    pythonize_bool(SANITIZER_COMMON_TEST_APPLE_TARGET_IS_HOST)

    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    # FIXME(dliew): LSan i386 on Darwin is completely broken right now.
    # so don't run the tests by default.
    # Tests fail on arm64, see https://github.com/llvm/llvm-project/issues/131678
    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin" AND
             ${tool} STREQUAL "lsan" AND
             (${arch} STREQUAL "i386" OR ${arch} MATCHES "arm64")))
      list(APPEND SANITIZER_COMMON_TESTSUITES
           ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
    endif()
  endforeach()
endforeach()

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-sanitizer-iossim-arm64" and similar. They also require an extra environment
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 16"
if(APPLE)
  set(SANITIZER_APPLE_PLATFORMS ${SANITIZER_COMMON_SUPPORTED_OS})

  foreach(platform ${SANITIZER_APPLE_PLATFORMS})
    if ("${platform}" STREQUAL "osx")
      # Skip macOS because it's handled by the code above that builds tests for the host machine.
      continue()
    endif()
    list_intersect(
      SANITIZER_COMMON_TEST_${platform}_ARCHS
      ALL_SANITIZER_COMMON_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
    )
    foreach(arch ${SANITIZER_COMMON_TEST_${platform}_ARCHS})
      set(SANITIZER_COMMON_TESTSUITES_${platform}_${arch})
      foreach(tool ${SUPPORTED_TOOLS})
        if ("${tool}" STREQUAL "lsan")
          continue()
        endif()

        set(SANITIZER_COMMON_LIT_TEST_MODE ${tool})

        get_test_cflags_for_apple_platform(
          "${platform}"
          "${arch}"
          SANITIZER_COMMON_TEST_TARGET_CFLAGS)
        set(CONFIG_NAME "${tool}-${arch}-${platform}")
        set(SANITIZER_COMMON_TEST_CONFIG_SUFFIX "-${arch}-${platform}")
        set(SANITIZER_COMMON_TEST_APPLE_PLATFORM "${platform}")
        set(SANITIZER_COMMON_TEST_APPLE_TARGET_IS_HOST OFF)
        pythonize_bool(SANITIZER_COMMON_TEST_APPLE_TARGET_IS_HOST)
        set(SANITIZER_COMMON_TEST_TARGET_ARCH "${arch}")
        set(SANITIZER_COMMON_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
        configure_lit_site_cfg(
          ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
          ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)

        list(APPEND SANITIZER_COMMON_TESTSUITES_${platform}_${arch}
             ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
      endforeach()

      add_lit_testsuite(check-sanitizer-${platform}-${arch}
        "sanitizer-common ${platform} ${arch} tests"
        ${SANITIZER_COMMON_TESTSUITES_${platform}_${arch}}
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS ${SANITIZER_COMMON_TEST_DEPS})
    endforeach()
  endforeach()
endif()


# Unit tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
# FIXME: support unit test in the android test runner
if (NOT ANDROID)
  list(APPEND SANITIZER_COMMON_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
  list(APPEND SANITIZER_COMMON_TEST_DEPS SanitizerUnitTests)
endif()

if(SANITIZER_COMMON_TESTSUITES)
  add_lit_testsuite(check-sanitizer "Running sanitizer_common tests"
    ${SANITIZER_COMMON_TESTSUITES}
    DEPENDS ${SANITIZER_COMMON_TEST_DEPS})
endif()
