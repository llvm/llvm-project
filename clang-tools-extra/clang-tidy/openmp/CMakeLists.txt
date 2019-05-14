set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyOpenMPModule
  ExceptionEscapeCheck.cpp
  OpenMPTidyModule.cpp
  UseDefaultNoneCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  clangTidyUtils
  )
