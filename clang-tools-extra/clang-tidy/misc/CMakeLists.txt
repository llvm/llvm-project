set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  AssertSideEffectCheck.cpp
  AssignOperatorSignatureCheck.cpp
  BoolPointerImplicitConversionCheck.cpp
  DefinitionsInHeadersCheck.cpp
  InaccurateEraseCheck.cpp
  InefficientAlgorithmCheck.cpp
  MacroParenthesesCheck.cpp
  MacroRepeatedSideEffectsCheck.cpp
  MiscTidyModule.cpp
  MoveConstantArgumentCheck.cpp 
  MoveConstructorInitCheck.cpp
  NewDeleteOverloadsCheck.cpp
  NoexceptMoveConstructorCheck.cpp
  NonCopyableObjects.cpp
  SizeofContainerCheck.cpp
  StaticAssertCheck.cpp
  StringIntegerAssignmentCheck.cpp
  SwappedArgumentsCheck.cpp
  ThrowByValueCatchByReferenceCheck.cpp
  UndelegatedConstructor.cpp
  UnusedAliasDeclsCheck.cpp
  UnusedParametersCheck.cpp
  UnusedRAIICheck.cpp
  UniqueptrResetReleaseCheck.cpp
  VirtualNearMissCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
