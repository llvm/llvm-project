# Multiple files define identically-named struct Lattice and class AnalysisImpl
# in anonymous namespaces, causing redefinition errors in unity builds.
set_source_files_properties(
  LoanPropagation.cpp
  MovedLoans.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangAnalysisLifetimeSafety
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

