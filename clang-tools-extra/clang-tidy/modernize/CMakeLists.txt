set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyModernizeModule
  DeprecatedHeadersCheck.cpp
  LoopConvertCheck.cpp
  LoopConvertUtils.cpp
  MakeSmartPtrCheck.cpp
  MakeSharedCheck.cpp
  MakeUniqueCheck.cpp
  ModernizeTidyModule.cpp
  PassByValueCheck.cpp
  RawStringLiteralCheck.cpp
  RedundantVoidArgCheck.cpp
  ReplaceAutoPtrCheck.cpp
  ShrinkToFitCheck.cpp
  UseAutoCheck.cpp
  UseDefaultCheck.cpp
  UseNullptrCheck.cpp
  UseOverrideCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils
  )
