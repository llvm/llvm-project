set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_clang_library(clangCrossTU
  CrossTranslationUnit.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangIndex
  )
