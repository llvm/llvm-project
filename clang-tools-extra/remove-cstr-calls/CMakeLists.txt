add_clang_executable(remove-cstr-calls
  RemoveCStrCalls.cpp
  )

target_link_libraries(remove-cstr-calls
  clangEdit clangTooling clangBasic clangAST clangASTMatchers)
