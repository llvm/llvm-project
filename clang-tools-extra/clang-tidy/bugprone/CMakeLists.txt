set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyBugproneModule
  ArgumentCommentCheck.cpp
  AssertSideEffectCheck.cpp
  BoolPointerImplicitConversionCheck.cpp
  BugproneTidyModule.cpp
  CopyConstructorInitCheck.cpp
  DanglingHandleCheck.cpp
  FoldInitTypeCheck.cpp
  ForwardDeclarationNamespaceCheck.cpp
  InaccurateEraseCheck.cpp
  IncorrectRoundingsCheck.cpp
  IntegerDivisionCheck.cpp
  MisplacedOperatorInStrlenInAllocCheck.cpp
  MoveForwardingReferenceCheck.cpp
  MultipleStatementMacroCheck.cpp
  StringConstructorCheck.cpp
  SuspiciousMemsetUsageCheck.cpp
  UndefinedMemoryManipulationCheck.cpp
  UseAfterMoveCheck.cpp
  VirtualNearMissCheck.cpp

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
