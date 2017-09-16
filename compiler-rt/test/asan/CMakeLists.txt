set(ASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ASAN_TESTSUITES)
set(ASAN_DYNAMIC_TESTSUITES)

# Before Windows 8 (CMAKE_SYSTEM_VERSION 6.2), reserving large regions of shadow
# memory allocated physical memory for page tables, which made it very
# unreliable. Remove the asan tests from check-all in this configuration.
set(SHADOW_MAPPING_UNRELIABLE FALSE)
if(OS_NAME MATCHES "Windows" AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
    ${CMAKE_SYSTEM_VERSION} LESS 6.2)
  set(SHADOW_MAPPING_UNRELIABLE TRUE)
  message(WARNING "Disabling ASan tests because they are unreliable on Windows 7 and earlier")
endif()

if (SHADOW_MAPPING_UNRELIABLE)
  set(EXCLUDE_FROM_ALL TRUE)
endif()

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "i386|arm|mips|mipsel")
    set(${bits} 32)
  elseif (${arch} MATCHES "x86_64|powerpc64|powerpc64le|aarch64|mips64|mips64el|s390x")
    set(${bits} 64)
  else()
    message(FATAL_ERROR "Unknown target architecture: ${arch}")
  endif()
endmacro()

set(ASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND ASAN_TEST_DEPS asan)
  if(NOT APPLE AND COMPILER_RT_HAS_LLD)
    list(APPEND ASAN_TEST_DEPS lld)
  endif()
endif()
set(ASAN_DYNAMIC_TEST_DEPS ${ASAN_TEST_DEPS})

set(ASAN_TEST_ARCH ${ASAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(ASAN_SUPPORTED_ARCH ASAN_TEST_ARCH)
endif()

foreach(arch ${ASAN_TEST_ARCH})
  if(ANDROID)
    set(ASAN_TEST_TARGET_ARCH ${arch}-android)
  else()
    set(ASAN_TEST_TARGET_ARCH ${arch})
  endif()

  set(ASAN_TEST_IOS "0")
  pythonize_bool(ASAN_TEST_IOS)
  set(ASAN_TEST_IOSSIM "0")
  pythonize_bool(ASAN_TEST_IOSSIM)

  string(TOLOWER "-${arch}-${OS_NAME}" ASAN_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} ASAN_TEST_BITS)
  get_test_cc_for_arch(${arch} ASAN_TEST_TARGET_CC ASAN_TEST_TARGET_CFLAGS)
  if(ANDROID)
    set(ASAN_TEST_DYNAMIC True)
  else()
    set(ASAN_TEST_DYNAMIC False)
  endif()
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
    )
  list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

  if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
    string(TOLOWER "-${arch}-${OS_NAME}-dynamic" ASAN_TEST_CONFIG_SUFFIX)
    set(ASAN_TEST_DYNAMIC True)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
    list(APPEND ASAN_DYNAMIC_TESTSUITES
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endif()
endforeach()

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-asan-iossim-x86_64" and similar. They also require that an extra env
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
if(APPLE)
  set(EXCLUDE_FROM_ALL ON)

  set(ASAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  set(ASAN_TEST_IOS "1")
  pythonize_bool(ASAN_TEST_IOS)
  set(ASAN_TEST_DYNAMIC True)

  foreach(arch ${DARWIN_iossim_ARCHS})
    set(ASAN_TEST_IOSSIM "1")
    pythonize_bool(ASAN_TEST_IOSSIM)
    set(ASAN_TEST_TARGET_ARCH ${arch})
    set(ASAN_TEST_TARGET_CFLAGS "-arch ${arch} -isysroot ${DARWIN_iossim_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
    set(ASAN_TEST_CONFIG_SUFFIX "-${arch}-iossim")
    get_bits_for_arch(${arch} ASAN_TEST_BITS)
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME "IOSSim${ARCH_UPPER_CASE}Config")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )
    add_lit_testsuite(check-asan-iossim-${arch} "AddressSanitizer iOS Simulator ${arch} tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${ASAN_TEST_DEPS})
  endforeach()

  foreach (arch ${DARWIN_ios_ARCHS})
    set(ASAN_TEST_IOSSIM "0")
    pythonize_bool(ASAN_TEST_IOSSIM)
    set(ASAN_TEST_TARGET_ARCH ${arch})
    set(ASAN_TEST_TARGET_CFLAGS "-arch ${arch} -isysroot ${DARWIN_ios_SYSROOT} ${COMPILER_RT_TEST_COMPILER_CFLAGS}")
    set(ASAN_TEST_CONFIG_SUFFIX "-${arch}-ios")
    get_bits_for_arch(${arch} ASAN_TEST_BITS)
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME "IOS${ARCH_UPPER_CASE}Config")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg
      )
    add_lit_testsuite(check-asan-ios-${arch} "AddressSanitizer iOS ${arch} tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
      DEPENDS ${ASAN_TEST_DEPS})
  endforeach()

  set(EXCLUDE_FROM_ALL OFF)
endif()

# Add unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  set(ASAN_TEST_DYNAMIC False)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
    set(ASAN_TEST_DYNAMIC True)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic/lit.site.cfg)
  endif()
  # FIXME: support unit test in the android test runner
  if (NOT ANDROID)
    list(APPEND ASAN_TEST_DEPS AsanUnitTests)
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
    if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
      list(APPEND ASAN_DYNAMIC_TEST_DEPS AsanDynamicUnitTests)
      list(APPEND ASAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic)
    endif()
  endif()
endif()

add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
  ${ASAN_TESTSUITES}
  DEPENDS ${ASAN_TEST_DEPS})
set_target_properties(check-asan PROPERTIES FOLDER "Compiler-RT Misc")

if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
  add_lit_testsuite(check-asan-dynamic
                    "Running the AddressSanitizer tests with dynamic runtime"
                    ${ASAN_DYNAMIC_TESTSUITES}
                    DEPENDS ${ASAN_DYNAMIC_TEST_DEPS})
  set_target_properties(check-asan-dynamic
                        PROPERTIES FOLDER "Compiler-RT Misc")
endif()

# Reset EXCLUDE_FROM_ALL to its initial value.
if (SHADOW_MAPPING_UNRELIABLE)
  set(EXCLUDE_FROM_ALL FALSE)
endif()
