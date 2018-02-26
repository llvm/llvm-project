set(MSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(MSAN_TESTSUITES)
set(MSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(MSAN_TEST_ARCH ${MSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(MSAN_SUPPORTED_ARCH MSAN_TEST_ARCH)
endif()

macro(add_msan_testsuite arch lld thinlto)
  set(MSAN_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} MSAN_TEST_TARGET_CC MSAN_TEST_TARGET_CFLAGS)

  string(TOUPPER ${arch} CONFIG_NAME)

  if (${thinlto})
    set(CONFIG_NAME "thinlto-${CONFIG_NAME}")
    list(APPEND MSAN_TEST_DEPS LTO)
  endif()
  if (${lld})
    set(CONFIG_NAME "lld-${CONFIG_NAME}")
    list(APPEND MSAN_TEST_DEPS lld)
  endif()
  set(MSAN_TEST_USE_THINLTO ${thinlto})
  set(MSAN_TEST_USE_LLD ${lld})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND MSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endmacro()

foreach(arch ${MSAN_TEST_ARCH})
  add_msan_testsuite(${arch} False False)

  if(COMPILER_RT_HAS_LLD AND arch STREQUAL "x86_64" AND NOT (APPLE OR WIN32))
    add_msan_testsuite(${arch} True False)
  endif()
endforeach()

if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND MSAN_TEST_DEPS msan)
endif()

if(COMPILER_RT_INCLUDE_TESTS AND COMPILER_RT_LIBCXX_PATH)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND MSAN_TEST_DEPS MsanUnitTests)
  list(APPEND MSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
endif()

add_lit_testsuite(check-msan "Running the MemorySanitizer tests"
  ${MSAN_TESTSUITES}
  DEPENDS ${MSAN_TEST_DEPS}
  )
set_target_properties(check-msan PROPERTIES FOLDER "Compiler-RT Misc")
