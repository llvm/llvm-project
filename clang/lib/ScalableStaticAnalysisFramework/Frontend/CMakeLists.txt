set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkFrontend
  TUSummaryExtractorFrontendAction.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangSema
  )
