set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkFrontend
  SourceTransformationFrontendAction.cpp
  TUSummaryExtractorFrontendAction.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangScalableStaticAnalysisFrameworkSourceTransformation
  clangSema
  clangToolingCore
  )
