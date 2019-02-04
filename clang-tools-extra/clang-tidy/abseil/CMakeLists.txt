set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyAbseilModule
  AbseilTidyModule.cpp
  DurationAdditionCheck.cpp
  DurationComparisonCheck.cpp
  DurationConversionCastCheck.cpp
  DurationDivisionCheck.cpp
  DurationFactoryFloatCheck.cpp
  DurationFactoryScaleCheck.cpp
  DurationRewriter.cpp
  DurationSubtractionCheck.cpp
  DurationUnnecessaryConversionCheck.cpp
  FasterStrsplitDelimiterCheck.cpp
  NoInternalDependenciesCheck.cpp
  NoNamespaceCheck.cpp
  RedundantStrcatCallsCheck.cpp
  StrCatAppendCheck.cpp
  StringFindStartswithCheck.cpp
  UpgradeDurationConversionsCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
