set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangFormat
  AffectedRangeManager.cpp
  BreakableToken.cpp
  ContinuationIndenter.cpp
  Format.cpp
  FormatToken.cpp
  FormatTokenLexer.cpp
  NamespaceEndCommentsFixer.cpp
  SortJavaScriptImports.cpp
  TokenAnalyzer.cpp
  TokenAnnotator.cpp
  UnwrappedLineFormatter.cpp
  UnwrappedLineParser.cpp
  UsingDeclarationsSorter.cpp
  WhitespaceManager.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangToolingCore
  )
