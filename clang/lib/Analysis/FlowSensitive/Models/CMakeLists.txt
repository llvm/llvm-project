add_clang_library(clangAnalysisFlowSensitiveModels
  ChromiumCheckModel.cpp
  DeadCodeModel.cpp
  UncheckedOptionalAccessModel.cpp

  LINK_LIBS
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  )
