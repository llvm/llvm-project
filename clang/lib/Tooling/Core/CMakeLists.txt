set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingCore
  Lookup.cpp
  Replacement.cpp
  QualTypeNames.cpp
  Diagnostic.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangRewrite
  )
