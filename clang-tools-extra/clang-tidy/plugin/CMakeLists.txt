add_clang_library(clangTidyPlugin
  ClangTidyPlugin.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSema
  clangTidy
  clangTooling
  ${ALL_CLANG_TIDY_CHECKS}
  )
