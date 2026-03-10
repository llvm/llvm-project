add_clang_library(clangAnalysisScalableFrontend
  TUSummaryExtractorFrontendAction.cpp

  LINK_LIBS
  clangAnalysisScalable
  clangAST
  clangBasic
  clangFrontend

  DEPENDS
  )
