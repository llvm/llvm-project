set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangTransformer
  RangeSelector.cpp
  RewriteRule.cpp
  SourceCode.cpp
  SourceCodeBuilders.cpp
  Stencil.cpp
  Transformer.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangToolingCore
  clangToolingRefactoring
  )
