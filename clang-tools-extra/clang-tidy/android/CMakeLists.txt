set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyAndroidModule
  AndroidTidyModule.cpp
  CloexecCreatCheck.cpp
  CloexecFopenCheck.cpp
  FileOpenFlagCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
