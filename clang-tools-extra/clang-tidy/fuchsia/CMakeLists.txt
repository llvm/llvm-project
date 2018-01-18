set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyFuchsiaModule
  DefaultArgumentsCheck.cpp
  FuchsiaTidyModule.cpp
  OverloadedOperatorCheck.cpp
  StaticallyConstructedObjectsCheck.cpp
  TrailingReturnCheck.cpp
  VirtualInheritanceCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
