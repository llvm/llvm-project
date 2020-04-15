set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyBoostModule
  BoostTidyModule.cpp
  UseToStringCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
