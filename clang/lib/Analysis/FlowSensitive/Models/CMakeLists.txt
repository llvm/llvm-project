add_clang_library(clangAnalysisFlowSensitiveModels
  UncheckedOptionalAccessModel.cpp

  LINK_LIBS
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  )
