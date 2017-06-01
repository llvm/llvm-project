set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UBSAN_TESTSUITES)
set(UBSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

macro(add_ubsan_testsuite test_mode sanitizer arch)
  set(UBSAN_LIT_TEST_MODE "${test_mode}")
  set(CONFIG_NAME ${UBSAN_LIT_TEST_MODE}-${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND UBSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND UBSAN_TEST_DEPS ${sanitizer})
  endif()
endmacro()

set(UBSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(UBSAN_SUPPORTED_ARCH UBSAN_TEST_ARCH)
endif()

foreach(arch ${UBSAN_TEST_ARCH})
  set(UBSAN_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} UBSAN_TEST_TARGET_CC UBSAN_TEST_TARGET_CFLAGS)
  add_ubsan_testsuite("Standalone" ubsan ${arch})

  if(COMPILER_RT_HAS_ASAN AND ";${ASAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
    # TODO(wwchrome): Re-enable ubsan for asan win 64-bit when ready.
    # Disable ubsan with AddressSanitizer tests for Windows 64-bit.
    if(NOT OS_NAME MATCHES "Windows" OR CMAKE_SIZEOF_VOID_P EQUAL 4)
      add_ubsan_testsuite("AddressSanitizer" asan ${arch})
    endif()
  endif()
  if(COMPILER_RT_HAS_MSAN AND ";${MSAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
    add_ubsan_testsuite("MemorySanitizer" msan ${arch})
  endif()
  if(COMPILER_RT_HAS_TSAN AND ";${TSAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
    add_ubsan_testsuite("ThreadSanitizer" tsan ${arch})
  endif()
endforeach()

add_lit_testsuite(check-ubsan "Running UndefinedBehaviorSanitizer tests"
  ${UBSAN_TESTSUITES}
  DEPENDS ${UBSAN_TEST_DEPS})
set_target_properties(check-ubsan PROPERTIES FOLDER "Compiler-RT Misc")
