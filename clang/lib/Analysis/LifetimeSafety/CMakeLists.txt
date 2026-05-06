add_clang_library(clangAnalysisLifetimeSafety
  AssignmentQuery.cpp
  Checker.cpp
  Facts.cpp
  FactsGenerator.cpp
  LifetimeAnnotations.cpp
  LifetimeSafety.cpp
  LiveOrigins.cpp
  LifetimeStats.cpp
  Loans.cpp
  LoanPropagation.cpp
  MovedLoans.cpp
  Origins.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  )

