if(LLVM_LIBC_FULL_BUILD)
  enable_testing()

  set(NEWHDGEN_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set(TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/newhdrgen/output)

  add_test(
    NAME newhdrgen_integration_test
    COMMAND python3 ${NEWHDGEN_TESTS_DIR}/test_integration.py --output_dir ${TEST_OUTPUT_DIR}
  )
 
  add_custom_target(check-newhdrgen
    COMMAND ${CMAKE_CTEST_COMMAND} -R newhdrgen_integration_test --output-on-failure
  )

  message(STATUS "Integration test for newhdrgen added.")
endif()
