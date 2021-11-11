set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UBSAN_TESTSUITES)
set(UBSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

macro(add_ubsan_testsuite test_mode sanitizer arch lld thinlto)
  set(UBSAN_LIT_TEST_MODE "${test_mode}")
  set(CONFIG_NAME ${UBSAN_LIT_TEST_MODE})
  if (${lld})
    set(CONFIG_NAME ${CONFIG_NAME}-lld)
    if ("lld" IN_LIST LLVM_ENABLE_PROJECTS)
      list(APPEND UBSAN_TEST_DEPS lld)
    endif()
  endif()
  if (${thinlto})
    set(CONFIG_NAME ${CONFIG_NAME}-thinlto)
    list(APPEND UBSAN_TEST_DEPS LTO)
  endif()
  set(UBSAN_TEST_USE_LLD ${lld})
  set(UBSAN_TEST_USE_THINLTO ${thinlto})
  set(CONFIG_NAME ${CONFIG_NAME}-${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND UBSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND UBSAN_TEST_DEPS ${sanitizer})
  endif()
endmacro()

macro(add_ubsan_testsuites test_mode sanitizer arch)
  add_ubsan_testsuite(${test_mode} ${sanitizer} ${arch} False False)

  if(COMPILER_RT_HAS_LLD AND arch STREQUAL "x86_64" AND NOT (APPLE OR WIN32))
    add_ubsan_testsuite(${test_mode} ${sanitizer} ${arch} True False)
  endif()
endmacro()

set(UBSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(UBSAN_SUPPORTED_ARCH UBSAN_TEST_ARCH)
endif()

foreach(arch ${UBSAN_TEST_ARCH})
  set(UBSAN_TEST_TARGET_ARCH ${arch})
  if (APPLE)
    set(UBSAN_TEST_APPLE_PLATFORM "osx")
    set(UBSAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")
  endif()
  get_test_cc_for_arch(${arch} UBSAN_TEST_TARGET_CC UBSAN_TEST_TARGET_CFLAGS)
  add_ubsan_testsuites("Standalone" ubsan ${arch})

  if(COMPILER_RT_HAS_ASAN AND ";${ASAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
    # TODO(wwchrome): Re-enable ubsan for asan win 64-bit when ready.
    # Disable ubsan with AddressSanitizer tests for Windows 64-bit,
    # 64-bit Solaris/x86, and SPARC.
    if((NOT (OS_NAME MATCHES "Windows" AND CMAKE_SIZEOF_VOID_P EQUAL 8)) AND
       (NOT (OS_NAME MATCHES "SunOS" AND ${arch} MATCHES x86_64)) AND
       (NOT ${arch} MATCHES sparc))
      add_ubsan_testsuites("AddressSanitizer" asan ${arch})
    endif()
  endif()
  if(COMPILER_RT_HAS_MSAN AND ";${MSAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
    add_ubsan_testsuites("MemorySanitizer" msan ${arch})
  endif()
  if(COMPILER_RT_HAS_TSAN AND ";${TSAN_SUPPORTED_ARCH};" MATCHES ";${arch};" AND NOT ANDROID)
    add_ubsan_testsuites("ThreadSanitizer" tsan ${arch})
  endif()
endforeach()

macro(add_ubsan_device_testsuite test_mode sanitizer platform arch)
  # Note we expect the caller to have already set UBSAN_TEST_TARGET_CFLAGS
  set(UBSAN_LIT_TEST_MODE "${test_mode}")
  set(CONFIG_NAME ${UBSAN_LIT_TEST_MODE}-${platform}-${arch})
  set(UBSAN_TEST_TARGET_ARCH ${arch})
  set(UBSAN_TEST_USE_LLD "False")
  set(UBSAN_TEST_USE_THINLTO "False")
  if (APPLE)
    set(UBSAN_TEST_APPLE_PLATFORM "${platform}")
    set(UBSAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
  else()
    unset(UBSAN_TEST_APPLE_PLATFORM)
    unset(UBSAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG)
  endif()
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  #list(APPEND UBSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND UBSAN_TEST_DEPS ${sanitizer})
  endif()
  add_lit_testsuite(check-ubsan-${test_mode}-${platform}-${arch}
    "UBSan ${CONFIG_NAME} tests"
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS ${UBSAN_TEST_DEPS})
endmacro()

if(APPLE)
  foreach(arch ${UBSAN_TEST_ARCH})
    set(UBSAN_TEST_TARGET_ARCH ${arch})
    get_test_cc_for_arch(${arch} UBSAN_TEST_TARGET_CC UBSAN_TEST_TARGET_CFLAGS)
    set(UBSAN_TEST_TARGET_CFLAGS "${UBSAN_TEST_TARGET_CFLAGS} -lc++abi")
    add_ubsan_testsuites("StandaloneStatic" ubsan ${arch})
  endforeach()

  # Device and simulator test suites.
  # These are not added into "check-all", in order to run these tests, use
  # "check-asan-iossim-x86_64" and similar. They also require that an extra env
  # variable to select which iOS device or simulator to use, e.g.:
  # SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
  set(UBSAN_APPLE_PLATFORMS ${UBSAN_SUPPORTED_OS})
  foreach(platform ${UBSAN_APPLE_PLATFORMS})
    list_intersect(
      UBSAN_TEST_${platform}_ARCHS
      UBSAN_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
      )
    foreach(arch ${UBSAN_TEST_${platform}_ARCHS})
      get_test_cflags_for_apple_platform(
        "${platform}"
        "${arch}"
        UBSAN_TEST_TARGET_CFLAGS
      )
      add_ubsan_device_testsuite("Standalone" ubsan ${platform} ${arch})

      if(COMPILER_RT_HAS_ASAN AND ";${ASAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
        add_ubsan_device_testsuite("AddressSanitizer" asan ${platform} ${arch})
      endif()

      if(COMPILER_RT_HAS_TSAN AND ";${TSAN_SUPPORTED_ARCH};" MATCHES ";${arch};")
        add_ubsan_device_testsuite("ThreadSanitizer" tsan ${platform} ${arch})
      endif()
    endforeach()
  endforeach()
endif()

add_lit_testsuite(check-ubsan "Running UndefinedBehaviorSanitizer tests"
  ${UBSAN_TESTSUITES}
  DEPENDS ${UBSAN_TEST_DEPS})
set_target_properties(check-ubsan PROPERTIES FOLDER "Compiler-RT Misc")

