set(NEWHDGEN_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

add_test(
  NAME newhdrgen_integration_test
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} python3 ${NEWHDGEN_TESTS_DIR}/test_integration.py
)

add_custom_target(check-newhdrgen
  COMMAND ${CMAKE_CTEST_COMMAND} -R newhdrgen_integration_test
)
#will remove once configuration done
message(STATUS "Integration test for newhdrgen added.")
