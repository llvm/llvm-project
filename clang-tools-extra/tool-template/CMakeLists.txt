set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(tool-template
  ToolTemplate.cpp
  )

clang_target_link_libraries(tool-template
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingRefactoring
  )
