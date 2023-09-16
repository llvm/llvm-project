set(ASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ASAN_TESTSUITES)
set(ASAN_DYNAMIC_TESTSUITES)

# Before Windows 8 (CMAKE_SYSTEM_VERSION 6.2), reserving large regions of shadow
# memory allocated physical memory for page tables, which made it very
# unreliable. Remove the asan tests from check-all in this configuration.
set(SHADOW_MAPPING_UNRELIABLE FALSE)
if(OS_NAME MATCHES "Windows" AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
    CMAKE_SYSTEM_VERSION LESS 6.2)
  set(SHADOW_MAPPING_UNRELIABLE TRUE)
  message(WARNING "Disabling ASan tests because they are unreliable on Windows 7 and earlier")
endif()

# Compile and run AMDGPU device address sanitizer tests only when
# -DSANITIZER_AMDGPU=1 is enabled.
if (SANITIZER_AMDGPU)
  set(SUPPORT_OFFLOAD_TESTS "true")
else()
  set(SUPPORT_OFFLOAD_TESTS "false")
endif()

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "x86_64|powerpc64|powerpc64le|aarch64|arm64|mips64|mips64el|s390x|sparcv9|riscv64|loongarch64")
    set(${bits} 64)
  elseif (${arch} MATCHES "i386|arm|mips|mipsel|sparc")
    set(${bits} 32)
  else()
    message(FATAL_ERROR "Unknown target architecture: ${arch}")
  endif()
endmacro()

set(ASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND ASAN_TEST_DEPS asan)
if(NOT COMPILER_RT_STANDALONE_BUILD)
  if(NOT APPLE AND COMPILER_RT_HAS_LLD AND "lld" IN_LIST LLVM_ENABLE_PROJECTS)
    list(APPEND ASAN_TEST_DEPS lld)
  endif()
endif()
set(ASAN_DYNAMIC_TEST_DEPS ${ASAN_TEST_DEPS})

set(ASAN_TEST_ARCH ${ASAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(ASAN_SUPPORTED_ARCH ASAN_TEST_ARCH)
endif()
list(REMOVE_ITEM ASAN_TEST_ARCH sparc sparcv9)
if(OS_NAME MATCHES "SunOS")
  list(REMOVE_ITEM ASAN_TEST_ARCH x86_64)
endif()

foreach(arch ${ASAN_TEST_ARCH})
  set(ASAN_TEST_TARGET_ARCH ${arch})
  set(ASAN_TEST_APPLE_PLATFORM "osx")
  set(ASAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")
  string(TOLOWER "-${arch}-${OS_NAME}" ASAN_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} ASAN_TEST_BITS)
  get_test_cc_for_arch(${arch} ASAN_TEST_TARGET_CC ASAN_TEST_TARGET_CFLAGS)
  if(ANDROID OR APPLE)
    set(ASAN_TEST_DYNAMIC True)
  else()
    set(ASAN_TEST_DYNAMIC False)
  endif()
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  if(NOT MINGW)
    # MinGW environments don't provide a statically linked CRT, so only the
    # dynamic asan test configuration can be expected to work.
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endif()

  if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
    string(TOLOWER "-${arch}-${OS_NAME}-dynamic" ASAN_TEST_CONFIG_SUFFIX)
    set(ASAN_TEST_DYNAMIC True)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND ASAN_DYNAMIC_TESTSUITES
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endif()

  # ARM on Linux may include thumb code, where fast unwinding does not work.
  # Enable unwind tables so that we do not end up falling back to it in tests.
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" AND NOT ANDROID AND ${arch} MATCHES "arm")
    set(ASAN_TEST_TARGET_CFLAGS "${ASAN_TEST_TARGET_CFLAGS} -funwind-tables")
  endif()
endforeach()

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-asan-iossim-x86_64" and similar. They also require that an extra env
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
if(APPLE)
  set(ASAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  set(ASAN_TEST_DYNAMIC True)
  set(ASAN_APPLE_PLATFORMS ${SANITIZER_COMMON_SUPPORTED_OS})

  foreach(platform ${ASAN_APPLE_PLATFORMS})
    if ("${platform}" STREQUAL "osx")
      # Skip macOS because it's handled by the code above that builds tests for the host machine.
      continue()
    endif()
    list_intersect(
      ASAN_TEST_${platform}_ARCHS
      ASAN_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
    )
    foreach(arch ${ASAN_TEST_${platform}_ARCHS})
      get_test_cflags_for_apple_platform(
        "${platform}"
        "${arch}"
        ASAN_TEST_TARGET_CFLAGS
        )
      string(TOUPPER "${arch}" ARCH_UPPER_CASE)
      get_capitalized_apple_platform("${platform}" PLATFORM_CAPITALIZED)
      set(CONFIG_NAME "${PLATFORM_CAPITALIZED}${ARCH_UPPER_CASE}Config")
      set(ASAN_TEST_CONFIG_SUFFIX "-${arch}-${platform}")
      set(ASAN_TEST_APPLE_PLATFORM "${platform}")
      set(ASAN_TEST_TARGET_ARCH "${arch}")
      set(ASAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
      get_bits_for_arch(${arch} ASAN_TEST_BITS)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
        )
      add_lit_testsuite(check-asan-${platform}-${arch} "AddressSanitizer ${platform} ${arch} tests"
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS ${ASAN_TEST_DEPS})
    endforeach()
  endforeach()
endif()

# Add unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  foreach(arch ${ASAN_TEST_ARCH})
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
    set(CONFIG_NAME_DYNAMIC ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig)

    if(NOT MINGW)
      # MinGW environments don't provide a statically linked CRT, so only the
      # dynamic asan test configuration can be expected to work.
      set(ASAN_TEST_DYNAMIC False)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
    endif()
    if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
      set(ASAN_TEST_DYNAMIC True)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME_DYNAMIC}/lit.site.cfg.py)
    endif()
    # FIXME: support unit test in the android test runner
    if (NOT ANDROID)
      if (NOT MINGW)
        list(APPEND ASAN_TEST_DEPS AsanUnitTests)
        list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
      endif()
      if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
        list(APPEND ASAN_DYNAMIC_TEST_DEPS AsanDynamicUnitTests)
        list(APPEND ASAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME_DYNAMIC})
      endif()
    endif()
  endforeach()
endif()

if (SHADOW_MAPPING_UNRELIABLE)
  set(exclude_from_check_all.g "EXCLUDE_FROM_CHECK_ALL")
else()
  set(exclude_from_check_all.g "")
endif()

add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
  ${ASAN_TESTSUITES}
  ${exclude_from_check_all}
  DEPENDS ${ASAN_TEST_DEPS})
set_target_properties(check-asan PROPERTIES FOLDER "Compiler-RT Misc")

if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
  add_lit_testsuite(check-asan-dynamic
                    "Running the AddressSanitizer tests with dynamic runtime"
                    ${ASAN_DYNAMIC_TESTSUITES}
                    ${exclude_from_check_all.g}
                    DEPENDS ${ASAN_DYNAMIC_TEST_DEPS})
  set_target_properties(check-asan-dynamic
                        PROPERTIES FOLDER "Compiler-RT Misc")
endif()
