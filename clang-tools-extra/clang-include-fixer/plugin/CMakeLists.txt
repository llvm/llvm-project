add_clang_library(clangIncludeFixerPlugin STATIC
  IncludeFixerPlugin.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangIncludeFixer
  clangParse
  clangSema
  clangTooling
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  omp_gen
  ClangDriverOptions
  )
