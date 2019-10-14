set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyUtils
  ASTUtils.cpp
  DeclRefExprUtils.cpp
  ExceptionAnalyzer.cpp
  ExprSequence.cpp
  FixItHintUtils.cpp
  HeaderFileExtensionsUtils.cpp
  HeaderGuard.cpp
  IncludeInserter.cpp
  IncludeSorter.cpp
  LexerUtils.cpp
  NamespaceAliaser.cpp
  OptionsUtils.cpp
  TransformerClangTidyCheck.cpp
  TypeTraits.cpp
  UsingInserter.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTransformer
  )
