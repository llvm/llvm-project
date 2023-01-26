if(CAN_TARGET_x86_64)
  set(TEST_ARCH ${X86_64})

  set(METADATA_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(METADATA_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

  set(METADATA_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
  if (NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND METADATA_TEST_DEPS asan ubsan)
  endif()

  set(SANITIZER_COMMON_TEST_TARGET_ARCH ${X86_64})
  get_test_cc_for_arch(x86_64 METADATA_TEST_TARGET_CC METADATA_TEST_TARGET_CFLAGS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

  add_lit_testsuite(check-sanmd "Running the SanitizerBinaryMetadata tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${METADATA_TEST_DEPS})
  set_target_properties(check-sanmd PROPERTIES FOLDER "Compiler-RT Misc")
endif()
