add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp
  WatchedLiteralsSolver.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  )
