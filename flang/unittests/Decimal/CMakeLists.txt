set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_flang_nongtest_unittest(quick-sanity-test
  FortranDecimal
)

# This test is not run by default as it takes a long time to execute.
add_flang_nongtest_unittest(thorough-test
  SLOW_TEST
  FortranDecimal
)
