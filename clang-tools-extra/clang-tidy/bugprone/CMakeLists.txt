set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyBugproneModule
  ArgumentCommentCheck.cpp
  BugproneTidyModule.cpp
  CopyConstructorInitCheck.cpp
  DanglingHandleCheck.cpp
  IntegerDivisionCheck.cpp
  MisplacedOperatorInStrlenInAllocCheck.cpp
  StringConstructorCheck.cpp
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
