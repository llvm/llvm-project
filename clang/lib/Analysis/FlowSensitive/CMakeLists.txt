add_clang_library(clangAnalysisFlowSensitive
  ControlFlowContext.cpp
  DataflowAnalysisContext.cpp
  DataflowEnvironment.cpp
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
