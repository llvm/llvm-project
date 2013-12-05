configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if(COMPILER_RT_CAN_EXECUTE_TESTS AND CAN_TARGET_x86_64)
  # Run TSan output tests only if we're sure we can produce working binaries.
  set(TSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${TSAN_RUNTIME_LIBRARIES})
  set(TSAN_TEST_PARAMS
    tsan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )
  if(LLVM_INCLUDE_TESTS)
    list(APPEND TSAN_TEST_DEPS TsanUnitTests)
  endif()
  add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${TSAN_TEST_PARAMS}
    DEPENDS ${TSAN_TEST_DEPS}
    )
  set_target_properties(check-tsan PROPERTIES FOLDER "TSan unittests")
elseif(LLVM_INCLUDE_TESTS AND CAN_TARGET_x86_64)
  # Otherwise run only TSan unit tests (they are linked using the
  # host compiler).
  add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Unit
    DEPENDS TsanUnitTests llvm-symbolizer)
  set_target_properties(check-tsan PROPERTIES FOLDER "TSan unittests")
endif()
