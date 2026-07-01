set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  SARIFTransformationReportFormat.cpp
  TransformationRegistry.cpp
  TransformationReportFormatRegistry.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangScalableStaticAnalysisCore
  clangTooling
  clangToolingCore
  )
