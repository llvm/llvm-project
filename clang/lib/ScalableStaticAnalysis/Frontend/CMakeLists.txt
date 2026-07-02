set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_clang_library(clangScalableStaticAnalysisFrontend
  TUSummaryExtractorFrontendAction.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangScalableStaticAnalysisAnalyses
  clangScalableStaticAnalysisCore
  clangSema
  )
