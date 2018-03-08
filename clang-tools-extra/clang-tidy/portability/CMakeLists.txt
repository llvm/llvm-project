set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyPortabilityModule
  PortabilityTidyModule.cpp
  SIMDIntrinsicsCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
