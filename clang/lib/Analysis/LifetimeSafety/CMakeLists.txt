add_clang_library(clangAnalysisLifetimeSafety
  LifetimeAnnotations.cpp
  LifetimeSafety.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex

  DEPENDS
  omp_gen
  ClangDriverOptions
  )
