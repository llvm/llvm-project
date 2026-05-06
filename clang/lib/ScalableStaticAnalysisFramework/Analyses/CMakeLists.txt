set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkAnalyses
  CallGraph/CallGraphExtractor.cpp
  CallGraph/CallGraphJSONFormat.cpp
  EntityPointerLevel/EntityPointerLevel.cpp
  EntityPointerLevel/EntityPointerLevelFormat.cpp
  PointerFlow/PointerFlow.cpp
  PointerFlow/PointerFlowAnalysis.cpp
  PointerFlow/PointerFlowExtractor.cpp
  PointerFlow/PointerFlowFormat.cpp
  SSAFAnalysesCommon.cpp
  UnsafeBufferUsage/UnsafeBufferUsage.cpp
  UnsafeBufferUsage/UnsafeBufferUsageAnalysis.cpp
  UnsafeBufferUsage/UnsafeBufferUsageExtractor.cpp
  UnsafeBufferUsage/UnsafeBufferUsageFormat.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangScalableStaticAnalysisFrameworkCore

  DEPENDS
  )
