add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  DataflowEnvironment.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  )
