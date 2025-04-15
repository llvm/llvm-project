set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangRewrite
  HTMLRewrite.cpp
  Rewriter.cpp
  TokenRewriter.cpp

  LINK_LIBS
  clangBasic
  clangLex
  )
