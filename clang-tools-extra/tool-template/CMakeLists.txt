add_clang_executable(tool-template
  ToolTemplate.cpp
  )

target_link_libraries(tool-template
  clangEdit clangTooling clangBasic clangAST clangASTMatchers)
