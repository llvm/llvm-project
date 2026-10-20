if(LLVM_LIBC_FULL_BUILD)
  enable_testing()

  set(HDRGEN_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set(TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/hdrgen/output)

  add_test(
    NAME hdrgen_integration_test
    COMMAND python3 ${HDRGEN_TESTS_DIR}/test_integration.py --output_dir ${TEST_OUTPUT_DIR}
  )

  add_custom_target(check-hdrgen
    COMMAND ${CMAKE_CTEST_COMMAND} -R hdrgen_integration_test --output-on-failure
  )

  message(STATUS "Integration test for hdrgen added.")
endif()
