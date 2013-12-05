set(MSAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(MSAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

if(MSAN_CAN_INSTRUMENT_LIBCXX)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
endif()

if(COMPILER_RT_CAN_EXECUTE_TESTS AND CAN_TARGET_x86_64)
  # Run MSan tests only if we're sure we may produce working binaries.
  set(MSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${MSAN_RUNTIME_LIBRARIES}
    msan_blacklist)
  set(MSAN_TEST_PARAMS
    msan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
  if(LLVM_INCLUDE_TESTS AND MSAN_CAN_INSTRUMENT_LIBCXX)
    list(APPEND MSAN_TEST_DEPS MsanUnitTests)
  endif()
  add_lit_testsuite(check-msan "Running the MemorySanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${MSAN_TEST_PARAMS}
    DEPENDS ${MSAN_TEST_DEPS}
    )
  set_target_properties(check-msan PROPERTIES FOLDER "MSan tests")
endif()
