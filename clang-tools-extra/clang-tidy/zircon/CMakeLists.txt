set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyZirconModule
  TemporaryObjectsCheck.cpp
  ZirconTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
