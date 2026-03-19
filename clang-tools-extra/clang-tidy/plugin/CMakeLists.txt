add_clang_library(clangTidyPlugin STATIC
  ClangTidyPlugin.cpp

  LINK_LIBS
  clangTidy
  ${ALL_CLANG_TIDY_CHECKS}

  DEPENDS
  omp_gen
  ClangDriverOptions
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
