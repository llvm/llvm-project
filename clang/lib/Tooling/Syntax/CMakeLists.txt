set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingSyntax
  Tokens.cpp

  LINK_LIBS
  clangBasic
  clangFrontend
  clangLex
  )
