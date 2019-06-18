set(TSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "x86_64")
  list(APPEND TSAN_TEST_DEPS GotsanRuntimeCheck)
endif()
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND TSAN_TEST_DEPS tsan)
endif()
if(COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH AND
   COMPILER_RT_TEST_COMPILER_ID STREQUAL "Clang"
   AND NOT APPLE AND NOT ANDROID)
  list(APPEND TSAN_TEST_DEPS libcxx_tsan)
  set(TSAN_HAS_LIBCXX True)
else()
  set(TSAN_HAS_LIBCXX False)
endif()

set(TSAN_TESTSUITES)

set(TSAN_TEST_ARCH ${TSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(TSAN_SUPPORTED_ARCH TSAN_TEST_ARCH)
endif()

foreach(arch ${TSAN_TEST_ARCH})
  set(TSAN_TEST_APPLE_PLATFORM "osx")

  set(TSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" TSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} TSAN_TEST_TARGET_CC TSAN_TEST_TARGET_CFLAGS)

  string(REPLACE ";" " " LIBDISPATCH_CFLAGS_STRING " ${COMPILER_RT_TEST_LIBDISPATCH_CFLAGS}")
  string(APPEND TSAN_TEST_TARGET_CFLAGS ${LIBDISPATCH_CFLAGS_STRING})

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND TSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-tsan-iossim-x86_64" and similar. They also require an extra environment
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
if(APPLE)
  # FIXME(dliew): This logic should be refactored to the way UBSan Darwin
  # testing is done.
  set(EXCLUDE_FROM_ALL ON)

  set(TSAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})

  list_intersect(TSAN_TEST_IOSSIM_ARCHS TSAN_SUPPORTED_ARCH DARWIN_iossim_ARCHS)
  foreach(arch ${TSAN_TEST_IOSSIM_ARCHS})
    set(TSAN_TEST_APPLE_PLATFORM "iossim")
    set(TSAN_TEST_TARGET_ARCH ${arch})
    get_test_cflags_for_apple_platform(
      "${TSAN_TEST_APPLE_PLATFORM}"
      "${TSAN_TEST_TARGET_ARCH}"
      TSAN_TEST_TARGET_CFLAGS
      )
    set(TSAN_TEST_CONFIG_SUFFIX "-${arch}-${TSAN_TEST_APPLE_PLATFORM}")
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME "IOSSim${ARCH_UPPER_CASE}Config")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )
    add_lit_testsuite(check-tsan-iossim-${arch} "ThreadSanitizer iOS Simulator ${arch} tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${TSAN_TEST_DEPS})
  endforeach()

  list_intersect(TSAN_TEST_IOS_ARCHS TSAN_SUPPORTED_ARCH DARWIN_ios_ARCHS)
  foreach(arch ${TSAN_TEST_IOS_ARCHS})
    set(TSAN_TEST_APPLE_PLATFORM "ios")
    set(TSAN_TEST_TARGET_ARCH ${arch})
    get_test_cflags_for_apple_platform(
      "${TSAN_TEST_APPLE_PLATFORM}"
      "${TSAN_TEST_TARGET_ARCH}"
      TSAN_TEST_TARGET_CFLAGS
      )
    set(TSAN_TEST_CONFIG_SUFFIX "-${arch}-${TSAN_TEST_APPLE_PLATFORM}")
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME "IOS${ARCH_UPPER_CASE}Config")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )
    add_lit_testsuite(check-tsan-ios-${arch} "ThreadSanitizer iOS ${arch} tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${TSAN_TEST_DEPS})
  endforeach()

  set(EXCLUDE_FROM_ALL OFF)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND TSAN_TEST_DEPS TsanUnitTests)
  list(APPEND TSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
endif()

add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
  ${TSAN_TESTSUITES}
  DEPENDS ${TSAN_TEST_DEPS})
set_target_properties(check-tsan PROPERTIES FOLDER "Compiler-RT Tests")
