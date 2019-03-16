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

  string(REPLACE ";" " " LIBDISPATCH_LINK_FLAGS_STRING " ${COMPILER_RT_INTERCEPT_LIBDISPATCH_LINK_FLAGS}")
  string(APPEND TSAN_TEST_TARGET_CFLAGS ${LIBDISPATCH_LINK_FLAGS_STRING})

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
  set(EXCLUDE_FROM_ALL ON)

  set(TSAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})

  set(TSAN_TEST_APPLE_PLATFORM "iossim")
  set(arch "x86_64")
  set(TSAN_TEST_TARGET_ARCH ${arch})
  set(TSAN_TEST_TARGET_CFLAGS "-arch ${arch} -isysroot ${DARWIN_iossim_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
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

  set(TSAN_TEST_APPLE_PLATFORM "ios")
  set(arch "arm64")
  set(TSAN_TEST_TARGET_ARCH ${arch})
  set(TSAN_TEST_TARGET_CFLAGS "-arch ${arch} -isysroot ${DARWIN_ios_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
  set(TSAN_TEST_CONFIG_SUFFIX "-${arch}-${TSAN_TEST_APPLE_PLATFORM}")
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME "IOS${ARCH_UPPER_CASE}Config")
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
    )
  add_lit_testsuite(check-tsan-ios-${arch} "ThreadSanitizer iOS Simulator ${arch} tests"
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
    DEPENDS ${TSAN_TEST_DEPS})

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
