set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  MisplacedConstCheck.cpp
  UnconventionalAssignOperatorCheck.cpp
  DefinitionsInHeadersCheck.cpp
  MacroParenthesesCheck.cpp
  MiscTidyModule.cpp
  NewDeleteOverloadsCheck.cpp
  NonCopyableObjects.cpp
  RedundantExpressionCheck.cpp
  SizeofContainerCheck.cpp
  SizeofExpressionCheck.cpp
  StaticAssertCheck.cpp
  ThrowByValueCatchByReferenceCheck.cpp
  UniqueptrResetReleaseCheck.cpp
  UnusedAliasDeclsCheck.cpp
  UnusedParametersCheck.cpp
  UnusedRAIICheck.cpp
  UnusedUsingDeclsCheck.cpp

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
