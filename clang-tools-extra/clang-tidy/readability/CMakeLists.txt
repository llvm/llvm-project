set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyReadabilityModule
  BracesAroundStatementsCheck.cpp
  ContainerSizeEmptyCheck.cpp
  ElseAfterReturnCheck.cpp
  FunctionSizeCheck.cpp
  IdentifierNamingCheck.cpp
  ImplicitBoolCastCheck.cpp
  InconsistentDeclarationParameterNameCheck.cpp
  NamedParameterCheck.cpp
  NamespaceCommentCheck.cpp
  ReadabilityTidyModule.cpp
  RedundantControlFlowCheck.cpp
  RedundantStringCStrCheck.cpp
  RedundantSmartptrGetCheck.cpp
  SimplifyBooleanExprCheck.cpp
  UniqueptrDeleteReleaseCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTooling
  )
