
add_clang_executable(clang-ast-dump
  ASTSrcLocProcessor.cpp
  ClangSrcLocDump.cpp
)

target_link_libraries(clang-ast-dump
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDriver
  clangFrontend
  clangSerialization
  clangToolingCore
)
