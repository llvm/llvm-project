add_clang_library(clangAnalysisFlowSensitiveModels
  ChromiumCheckModel.cpp
  NullPointerAnalysisModel.cpp
  UncheckedOptionalAccessModel.cpp
  UncheckedStatusOrAccessModel.cpp

  LINK_LIBS
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  )
