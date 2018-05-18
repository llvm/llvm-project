set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingCore
  Diagnostic.cpp
  Lookup.cpp
  Replacement.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangRewrite
  )
