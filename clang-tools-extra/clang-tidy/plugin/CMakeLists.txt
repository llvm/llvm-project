add_clang_library(clangTidyPlugin
  ClangTidyPlugin.cpp

  LINK_LIBS
  clangTidy
  ${ALL_CLANG_TIDY_CHECKS}
  )

clang_target_link_libraries(clangTidyPlugin
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSema
  clangTooling
  )
