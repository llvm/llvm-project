set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_executable(quick-sanity-test
  quick-sanity-test.cpp
)

target_link_libraries(quick-sanity-test
  FortranDecimal
  LLVMSupport
)

add_executable(thorough-test
  thorough-test.cpp
)

target_link_libraries(thorough-test
  FortranDecimal
  LLVMSupport
)

add_test(NAME Sanity COMMAND quick-sanity-test)
