set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangMove
  ClangMove.cpp
  HelperDeclRefGraph.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
