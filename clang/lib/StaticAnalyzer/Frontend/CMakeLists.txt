include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../Checkers )

set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangStaticAnalyzerFrontend
  AnalysisConsumer.cpp
  AnalyzerHelpFlags.cpp
  CheckerRegistry.cpp
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
  clangRewrite
  clangToolingCore
  )
