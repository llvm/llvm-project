set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyPerformanceModule
  ForRangeCopyCheck.cpp
  ImplicitCastInLoopCheck.cpp
  PerformanceTidyModule.cpp
  UnnecessaryCopyInitialization.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
