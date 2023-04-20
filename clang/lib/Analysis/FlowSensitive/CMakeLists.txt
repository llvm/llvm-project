add_clang_library(clangAnalysisFlowSensitive
  Arena.cpp
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
  Logger.cpp
  Transfer.cpp
  TypeErasedDataflowAnalysis.cpp
  Value.cpp
  WatchedLiteralsSolver.cpp
  DebugSupport.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  )

add_subdirectory(Models)
