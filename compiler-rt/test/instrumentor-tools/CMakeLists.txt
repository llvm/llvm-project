set(INSTRUMENTOR_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTRUMENTOR_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Debug: Print all relevant variables
set(INSTRUMENTOR_TESTSUITES)
set(INSTRUMENTOR_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND INSTRUMENTOR_TEST_DEPS flop-counter)
list(APPEND INSTRUMENTOR_TEST_DEPS fp-precision-analysis)

# Check if INSTRUMENTOR_SUPPORTED_ARCH is defined
if(NOT DEFINED INSTRUMENTOR_SUPPORTED_ARCH)
  message(STATUS "INSTRUMENTOR_SUPPORTED_ARCH is NOT DEFINED - using fallback")
  # Use the same architectures as FLOP counter
  if(APPLE)
    set(INSTRUMENTOR_SUPPORTED_ARCH ${DARWIN_osx_ARCHS})
  else()
    set(INSTRUMENTOR_SUPPORTED_ARCH ${COMPILER_RT_DEFAULT_TARGET_ARCH})
  endif()
endif()

set(INSTRUMENTOR_TEST_ARCH ${INSTRUMENTOR_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(INSTRUMENTOR_SUPPORTED_ARCH INSTRUMENTOR_TEST_ARCH)
endif()


macro(add_instrumentor_testsuite test_mode sanitizer arch)
  set(INSTRUMENTOR_LIT_TEST_MODE "${test_mode}")
  set(CONFIG_NAME ${test_mode}-${arch})

  set(INSTRUMENTOR_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} INSTRUMENTOR_TEST_TARGET_CC INSTRUMENTOR_TEST_TARGET_CFLAGS)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND INSTRUMENTOR_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

endmacro()

if(INSTRUMENTOR_TEST_ARCH)
  foreach(arch ${INSTRUMENTOR_TEST_ARCH})
    add_instrumentor_testsuite("InstrumentorTools" instrumentor-tools ${arch})
  endforeach()
else()
  message(WARNING "No architectures configured for instrumentor-tools tests!")
endif()

if(INSTRUMENTOR_TESTSUITES)
  add_lit_testsuite(check-instrumentor-tools "Running the instrumentor-tools tests"
    ${INSTRUMENTOR_TESTSUITES}
    DEPENDS ${INSTRUMENTOR_TEST_DEPS})
else()
  message(WARNING "No test suites configured for instrumentor-tools!")
endif()
