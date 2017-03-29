set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyHICPPModule
  NoAssemblerCheck.cpp
  HICPPTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
