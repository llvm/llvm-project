set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkSourceTransformation
  TransformationRegistry.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangScalableStaticAnalysisFrameworkCore
  clangToolingCore
  )
