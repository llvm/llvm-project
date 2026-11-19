set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  EntityLookup.cpp
  SARIFTransformationReportFormat.cpp
  TransformationRegistry.cpp
  Transformations/CppBoundedBuffers.cpp
  YAMLSourceEditFormat.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangScalableStaticAnalysisCore
  clangToolingCore
  )
