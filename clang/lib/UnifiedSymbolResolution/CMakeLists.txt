set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangUnifiedSymbolResolution
  USRGeneration.cpp

  DEPENDS
  omp_gen

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )
