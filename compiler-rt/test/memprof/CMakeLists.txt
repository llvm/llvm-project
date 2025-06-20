set(MEMPROF_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(MEMPROF_TESTSUITES)
set(MEMPROF_DYNAMIC_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "x86_64|arm64")
    set(${bits} 64)
  else()
    message(FATAL_ERROR "Unexpected target architecture: ${arch}")
  endif()
endmacro()

set(MEMPROF_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} memprof)
if(NOT COMPILER_RT_STANDALONE_BUILD AND COMPILER_RT_HAS_LLD AND TARGET lld)
  list(APPEND MEMPROF_TEST_DEPS lld)
endif()
set(MEMPROF_DYNAMIC_TEST_DEPS ${MEMPROF_TEST_DEPS})

set(MEMPROF_TEST_ARCH ${MEMPROF_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(MEMPROF_SUPPORTED_ARCH MEMPROF_TEST_ARCH)
endif()

foreach(arch ${MEMPROF_TEST_ARCH})
  set(MEMPROF_TEST_TARGET_ARCH ${arch})
  set(MEMPROF_TEST_APPLE_PLATFORM "osx")
  set(MEMPROF_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")
  string(TOLOWER "-${arch}-${OS_NAME}" MEMPROF_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} MEMPROF_TEST_BITS)
  get_test_cc_for_arch(${arch} MEMPROF_TEST_TARGET_CC MEMPROF_TEST_TARGET_CFLAGS)
  if(APPLE)
    set(MEMPROF_TEST_DYNAMIC True)
  else()
    set(MEMPROF_TEST_DYNAMIC False)
  endif()
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    )
  list(APPEND MEMPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

  if (COMPILER_RT_MEMPROF_HAS_STATIC_RUNTIME)
    string(TOLOWER "-${arch}-${OS_NAME}-dynamic" MEMPROF_TEST_CONFIG_SUFFIX)
    set(MEMPROF_TEST_DYNAMIC True)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND MEMPROF_DYNAMIC_TESTSUITES
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endif()
endforeach()

# Add unit tests.
foreach(arch ${MEMPROF_TEST_ARCH})
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND MEMPROF_TEST_DEPS MemProfUnitTests)
  list(APPEND MEMPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-memprof-iossim-x86_64" and similar. They also require that an extra env
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
if(APPLE)
  set(MEMPROF_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  set(MEMPROF_TEST_DYNAMIC True)
  set(MEMPROF_APPLE_PLATFORMS ${SANITIZER_COMMON_SUPPORTED_OS})

  foreach(platform ${MEMPROF_APPLE_PLATFORMS})
    if ("${platform}" STREQUAL "osx")
      # Skip macOS because it's handled by the code above that builds tests for the host machine.
      continue()
    endif()
    list_intersect(
      MEMPROF_TEST_${platform}_ARCHS
      MEMPROF_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
    )
    foreach(arch ${MEMPROF_TEST_${platform}_ARCHS})
      get_test_cflags_for_apple_platform(
        "${platform}"
        "${arch}"
        MEMPROF_TEST_TARGET_CFLAGS
        )
      string(TOUPPER "${arch}" ARCH_UPPER_CASE)
      get_capitalized_apple_platform("${platform}" PLATFORM_CAPITALIZED)
      set(CONFIG_NAME "${PLATFORM_CAPITALIZED}${ARCH_UPPER_CASE}Config")
      set(MEMPROF_TEST_CONFIG_SUFFIX "-${arch}-${platform}")
      set(MEMPROF_TEST_APPLE_PLATFORM "${platform}")
      set(MEMPROF_TEST_TARGET_ARCH "${arch}")
      set(MEMPROF_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
      get_bits_for_arch(${arch} MEMPROF_TEST_BITS)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
        )
      add_lit_testsuite(check-memprof-${platform}-${arch} "AddressSanitizer ${platform} ${arch} tests"
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS ${MEMPROF_TEST_DEPS})
    endforeach()
  endforeach()
endif()

add_lit_testsuite(check-memprof "Running the MemProfiler tests"
  ${MEMPROF_TESTSUITES}
  DEPENDS ${MEMPROF_TEST_DEPS})

if (COMPILER_RT_MEMPROF_HAS_STATIC_RUNTIME)
  add_lit_testsuite(check-memprof-dynamic
    "Running the MemProfiler tests with dynamic runtime"
    ${MEMPROF_DYNAMIC_TESTSUITES}
    ${exclude_from_check_all.g}
    DEPENDS ${MEMPROF_DYNAMIC_TEST_DEPS})
endif()
