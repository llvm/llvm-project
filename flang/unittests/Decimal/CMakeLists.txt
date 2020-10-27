set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_flang_nongtest_unittest(quick-sanity-test
  FortranDecimal
  LLVMSupport
)

# This test is not run by default as it takes a long time to execute
add_executable(thorough-test
  thorough-test.cpp
)

target_link_libraries(thorough-test
  FortranDecimal
  LLVMSupport
)
