set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangFormat
  AffectedRangeManager.cpp
  BreakableToken.cpp
  ContinuationIndenter.cpp
  DefinitionBlockSeparator.cpp
  Format.cpp
  FormatToken.cpp
  FormatTokenLexer.cpp
  IntegerLiteralSeparatorFixer.cpp
  MacroCallReconstructor.cpp
  MacroExpander.cpp
  NamespaceEndCommentsFixer.cpp
  QualifierAlignmentFixer.cpp
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
  clangToolingInclusions
  )
