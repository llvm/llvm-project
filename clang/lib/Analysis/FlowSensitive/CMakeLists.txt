add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
  SourceLocationsLattice.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp
  WatchedLiteralsSolver.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  )
