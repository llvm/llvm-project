set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  TransformationRegistry.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangScalableStaticAnalysisCore
  clangToolingCore
  )
