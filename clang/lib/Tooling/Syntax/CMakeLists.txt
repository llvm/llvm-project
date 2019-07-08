set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingSyntax
  BuildTree.cpp
  Nodes.cpp
  Tokens.cpp
  Tree.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangToolingCore
  )
