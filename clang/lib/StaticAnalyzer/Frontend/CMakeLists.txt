include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../Checkers )

set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangStaticAnalyzerFrontend
  AnalysisConsumer.cpp
  CheckerRegistration.cpp
  FrontendActions.cpp
  ModelConsumer.cpp
  ModelInjector.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCrossTU
  clangFrontend
  clangLex
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  )
