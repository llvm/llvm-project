set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCERTModule
  CERTTidyModule.cpp
  CommandProcessorCheck.cpp
  FloatLoopCounter.cpp
  SetLongJmpCheck.cpp
  StaticObjectExceptionCheck.cpp
  StrToNumCheck.cpp
  ThrownExceptionTypeCheck.cpp
  VariadicFunctionDefCheck.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyUtils
  )
