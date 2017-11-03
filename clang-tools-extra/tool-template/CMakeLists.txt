set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(tool-template
  ToolTemplate.cpp
  )

target_link_libraries(tool-template
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingRefactor
  )
