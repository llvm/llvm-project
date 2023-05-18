set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyAbseilModule
  AbseilTidyModule.cpp
  CleanupCtadCheck.cpp
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
  StringFindStrContainsCheck.cpp
  TimeComparisonCheck.cpp
  TimeSubtractionCheck.cpp
  UpgradeDurationConversionsCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyAbseilModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  clangTransformer
  )
