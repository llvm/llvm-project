set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  TransformationRegistry.cpp
  YAMLSourceEditFormat.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangScalableStaticAnalysisCore
  clangToolingCore
  )
