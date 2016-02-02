set(TSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "x86_64")
  list(APPEND TSAN_TEST_DEPS GotsanRuntimeCheck)
endif()
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND TSAN_TEST_DEPS tsan)
endif()
if(COMPILER_RT_HAS_LIBCXX_SOURCES AND
   COMPILER_RT_TEST_COMPILER_ID STREQUAL "Clang"
   AND NOT APPLE)
  list(APPEND TSAN_TEST_DEPS libcxx_tsan)
  set(TSAN_HAS_LIBCXX True)
else()
  set(TSAN_HAS_LIBCXX False)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND TSAN_TEST_DEPS TsanUnitTests)
endif()

add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TSAN_TEST_DEPS})
set_target_properties(check-tsan PROPERTIES FOLDER "TSan tests")
