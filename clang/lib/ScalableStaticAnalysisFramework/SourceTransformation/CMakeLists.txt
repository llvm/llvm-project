set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkSourceTransformation
  SARIFTransformationReportFormat.cpp
  TransformationRegistry.cpp
  YAMLSourceEditFormat.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangScalableStaticAnalysisFrameworkCore
  clangToolingCore
  )
