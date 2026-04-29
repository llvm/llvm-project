set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkFrontend
  SourcePassAnalysisFrontendAction.cpp
  TUSummaryExtractorFrontendAction.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangSema
  )
