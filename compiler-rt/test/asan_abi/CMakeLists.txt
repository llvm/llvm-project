set(ASAN_ABI_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASAN_ABI_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "arm64|x86_64")
    set(${bits} 64)
  else()
    message(FATAL_ERROR "Unknown target architecture: ${arch}")
  endif()
endmacro()

set(ASAN_ABI_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} asan_abi)

set(ASAN_ABI_TEST_ARCH ${ASAN_ABI_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(ASAN_ABI_SUPPORTED_ARCH ASAN_ABI_TEST_ARCH)
endif()

foreach(arch ${ASAN_ABI_TEST_ARCH})
  set(ASAN_ABI_TEST_TARGET_ARCH ${arch})
  set(ASAN_ABI_TEST_APPLE_PLATFORM "osx")
  set(ASAN_ABI_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")
  string(TOLOWER "-${arch}-${OS_NAME}" ASAN_ABI_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} ASAN_ABI_TEST_BITS)
  get_test_cc_for_arch(${arch} ASAN_ABI_TEST_TARGET_CC ASAN_ABI_TEST_TARGET_CFLAGS)

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  list(APPEND ASAN_ABI_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    )
endforeach()

if(APPLE)
  set(ASAN_ABI_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  set(ASAN_ABI_TEST_APPLE_PLATFORMS ${SANITIZER_COMMON_SUPPORTED_OS})

  foreach(platform ${ASAN_ABI_TEST_APPLE_PLATFORMS})
    if ("${platform}" STREQUAL "osx")
      # Skip macOS because it's handled by the code above that builds tests for the host machine.
      continue()
    endif()
    list_intersect(
      ASAN_ABI_TEST_${platform}_ARCHS
      ASAN_ABI_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
    )
    foreach(arch ${ASAN_ABI_TEST_${platform}_ARCHS})
      get_test_cflags_for_apple_platform(
        "${platform}"
        "${arch}"
        ASAN_ABI_TEST_TARGET_CFLAGS
        )
      string(TOUPPER "${arch}" ARCH_UPPER_CASE)
      get_capitalized_apple_platform("${platform}" PLATFORM_CAPITALIZED)
      set(CONFIG_NAME "${PLATFORM_CAPITALIZED}${ARCH_UPPER_CASE}Config")
      set(ASAN_ABI_TEST_CONFIG_SUFFIX "-${arch}-${platform}")
      set(ASAN_ABI_TEST_APPLE_PLATFORM "${platform}")
      set(ASAN_ABI_TEST_TARGET_ARCH "${arch}")
      set(ASAN_ABI_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
      get_bits_for_arch(${arch} ASAN_ABI_TEST_BITS)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
        )
      add_lit_testsuite(check-asan-abi-${platform}-${arch} "AddressSanitizerABI ${platform} ${arch} tests"
        ${ASAN_ABI_TESTSUITES}
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS ${ASAN_ABI_TEST_DEPS})
    endforeach()
  endforeach()
endif()

add_lit_testsuite(check-asan-abi "Running the AddressSanitizerABI tests"
  ${ASAN_ABI_TESTSUITES}
  ${exclude_from_check_all}
  DEPENDS ${ASAN_ABI_TEST_DEPS})
set_target_properties(check-asan-abi PROPERTIES FOLDER "Compiler-RT Misc")
