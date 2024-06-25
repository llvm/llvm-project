set(MEMPROF_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(MEMPROF_TESTSUITES)
set(MEMPROF_DYNAMIC_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "x86_64")
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

foreach(arch ${MEMPROF_TEST_ARCH})
  set(MEMPROF_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" MEMPROF_TEST_CONFIG_SUFFIX)
  get_bits_for_arch(${arch} MEMPROF_TEST_BITS)
  get_test_cc_for_arch(${arch} MEMPROF_TEST_TARGET_CC MEMPROF_TEST_TARGET_CFLAGS)
  set(MEMPROF_TEST_DYNAMIC False)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    )
  list(APPEND MEMPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

  string(TOLOWER "-${arch}-${OS_NAME}-dynamic" MEMPROF_TEST_CONFIG_SUFFIX)
  set(MEMPROF_TEST_DYNAMIC True)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND MEMPROF_DYNAMIC_TESTSUITES
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

# Add unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  foreach(arch ${MEMPROF_TEST_ARCH})
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND MEMPROF_TEST_DEPS MemProfUnitTests)
    list(APPEND MEMPROF_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
  endforeach()
endif()

add_lit_testsuite(check-memprof "Running the MemProfiler tests"
  ${MEMPROF_TESTSUITES}
  DEPENDS ${MEMPROF_TEST_DEPS})

add_lit_testsuite(check-memprof-dynamic
	"Running the MemProfiler tests with dynamic runtime"
  ${MEMPROF_DYNAMIC_TESTSUITES}
  ${exclude_from_check_all.g}
  DEPENDS ${MEMPROF_DYNAMIC_TEST_DEPS})
