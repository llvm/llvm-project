add_clang_library(clangAnalysisFlowSensitiveTesting
  DataflowAnalysis.cpp

  LINK_LIBS
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  )
