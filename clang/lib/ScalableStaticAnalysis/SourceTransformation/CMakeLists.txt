set(LLVM_LINK_COMPONENTS
  Support
  )

# TODO: move SSAFAnalysesCommon.h to a higher-up folder shared by both
# libraries instead of reaching into a sibling library's private header.
#
# Needed to reach clang/lib/ScalableStaticAnalysis/Analyses/SSAFAnalysesCommon.h,
# a private header shared by the two libraries.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Analyses)

add_clang_library(clangScalableStaticAnalysisSourceTransformation
  SARIFTransformationReportFormat.cpp
  TransformationRegistry.cpp
  Transformations/CppBoundedBuffers.cpp
  YAMLSourceEditFormat.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangScalableStaticAnalysisAnalyses
  clangScalableStaticAnalysisCore
  clangToolingCore
  clangToolingRefactoring
  )
