set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyUtils
  DeclRefExprUtils.cpp
  FixItHintUtils.cpp
  HeaderFileExtensionsUtils.cpp
  HeaderGuard.cpp
  IncludeInserter.cpp
  IncludeSorter.cpp
  LexerUtils.cpp
  OptionsUtils.cpp
  TypeTraits.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
