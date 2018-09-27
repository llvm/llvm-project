set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMPIModule
  BufferDerefCheck.cpp
  MPITidyModule.cpp
  TypeMismatchCheck.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  clangStaticAnalyzerCheckers
  )
