set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  SARIFTransformationReportFormat.cpp
  TransformationRegistry.cpp
  Transformations/CppBoundedBuffers.cpp
  Transformations/NullabilityAnnotations.cpp
  YAMLSourceEditFormat.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangLex
  clangScalableStaticAnalysisCore
  clangToolingCore
  )
