set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyBugproneModule
  BugproneTidyModule.cpp
  CopyConstructorInitCheck.cpp
  IntegerDivisionCheck.cpp
  SuspiciousMemsetUsageCheck.cpp
  UndefinedMemoryManipulationCheck.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
