set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyUtils
  Aliasing.cpp
  ASTUtils.cpp
  BracesAroundStatement.cpp
  DeclRefExprUtils.cpp
  DesignatedInitializers.cpp
  ExceptionAnalyzer.cpp
  ExceptionSpecAnalyzer.cpp
  ExprSequence.cpp
  FileExtensionsUtils.cpp
  FormatStringConverter.cpp
  FixItHintUtils.cpp
  HeaderGuard.cpp
  IncludeInserter.cpp
  IncludeSorter.cpp
  LexerUtils.cpp
  Matchers.cpp
  NamespaceAliaser.cpp
  OptionsUtils.cpp
  RenamerClangTidyCheck.cpp
  TransformerClangTidyCheck.cpp
  TypeTraits.cpp
  UseRangesCheck.cpp
  UsingInserter.cpp

  LINK_LIBS
  clangTidy

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyUtils
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSema
  clangTooling
  clangTransformer
  )
