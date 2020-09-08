set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyAlteraModule
  AlteraTidyModule.cpp
  StructPackAlignCheck.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
