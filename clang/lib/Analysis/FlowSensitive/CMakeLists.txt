add_clang_library(clangAnalysisFlowSensitive
  TypeErasedDataflowAnalysis.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  )
