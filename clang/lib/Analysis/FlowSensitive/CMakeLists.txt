add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
  SourceLocationsLattice.cpp
  TestingSupport.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp
  WatchedLiteralsSolver.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  )

add_subdirectory(Models)
