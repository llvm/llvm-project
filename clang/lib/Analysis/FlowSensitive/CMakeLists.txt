add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  TypeErasedDataflowAnalysis.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  )
