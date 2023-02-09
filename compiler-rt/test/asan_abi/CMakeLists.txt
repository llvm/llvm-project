set(ASAN_ABI_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ASAN_ABI_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "arm64")
    set(${bits} 64)
  else()
    message(FATAL_ERROR "Unknown target architecture: ${arch}")
  endif()
endmacro()

set(ASAN_ABI_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND ASAN_ABI_TEST_DEPS asan_abi)
endif()
set(ASAN_ABI_DYNAMIC_TEST_DEPS ${ASAN_ABI_TEST_DEPS})

set(ASAN_ABI_TEST_ARCH arm64)

foreach(arch ${ASAN_ABI_TEST_ARCH})
  set(ASAN_ABI_TEST_TARGET_ARCH ${arch})
  set(ASAN_ABI_TEST_APPLE_PLATFORM "osx")
  set(ASAN_ABI_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")
  string(TOLOWER "-${arch}-${OS_NAME}" ASAN_ABI_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} ASAN_ABI_TEST_BITS)
  get_test_cc_for_arch(${arch} ASAN_ABI_TEST_TARGET_CC ASAN_ABI_TEST_TARGET_CFLAGS)
  if(ANDROID OR APPLE)
    set(ASAN_ABI_TEST_DYNAMIC True)
  else()
    set(ASAN_ABI_TEST_DYNAMIC False)
  endif()
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  list(APPEND ASAN_ABI_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    )
endforeach()

add_lit_testsuite(check-asan-abi "Running the AddressSanitizerABI tests"
  ${ASAN_ABI_TESTSUITES}
  ${exclude_from_check_all}
  DEPENDS ${ASAN_ABI_TEST_DEPS})
set_target_properties(check-asan-abi PROPERTIES FOLDER "Compiler-RT Misc")
