set(LLVM_LINK_COMPONENTS
  Support
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
