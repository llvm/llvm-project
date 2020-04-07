set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMModule
  HeaderGuardCheck.cpp
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  PreferIsaOrDynCastInConditionalsCheck.cpp
  PreferRegisterOverUnsignedCheck.cpp
  TwineLocalCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils
  clangTooling
  )
