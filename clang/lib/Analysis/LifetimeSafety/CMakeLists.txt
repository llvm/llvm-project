add_clang_library(clangAnalysisLifetimeSafety
  Checker.cpp
  Facts.cpp
  FactsGenerator.cpp
  LifetimeAnnotations.cpp
  LifetimeSafety.cpp
  LiveOrigins.cpp
  Loans.cpp
  LoanPropagation.cpp
  Origins.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  )

