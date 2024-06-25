set(NSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(NSAN_TESTSUITES)
set(NSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} nsan)

if(COMPILER_RT_INCLUDE_TESTS AND
   COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
  list(APPEND NSAN_TEST_DEPS NsanUnitTests)
  list(APPEND NSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
endif()

add_lit_testsuite(check-nsan "Running the numerical stability sanitizer tests"
  ${NSAN_TESTSUITES}
  DEPENDS ${NSAN_TEST_DEPS}
  )
