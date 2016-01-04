set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyUtils
  HeaderGuard.cpp
  IncludeInserter.cpp
  IncludeSorter.cpp
  LexerUtils.cpp
  TypeTraits.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
