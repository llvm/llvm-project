set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_clang_library(clangScalableStaticAnalysisAnalyses
  CallGraph/CallGraphExtractor.cpp
  CallGraph/CallGraphJSONFormat.cpp
  EntityPointerLevel/EntityPointerLevel.cpp
  EntityPointerLevel/EntityPointerLevelFormat.cpp
  PointerFlow/PointerFlow.cpp
  PointerFlow/PointerFlowAnalysis.cpp
  PointerFlow/PointerFlowExtractor.cpp
  PointerFlow/PointerFlowFormat.cpp
  SharedLexicalRepresentation/EntitySourceLocationExtractor.cpp
  SharedLexicalRepresentation/SharedLexicalRepresentationFormat.cpp
  SSAFAnalysesCommon.cpp
  TypeConstrainedPointers/TypeConstrainedPointers.cpp
  UnsafeBufferUsage/UnsafeBufferUsage.cpp
  UnsafeBufferUsage/UnsafeBufferUsageAnalysis.cpp
  UnsafeBufferUsage/UnsafeBufferUsageExtractor.cpp
  UnsafeBufferUsage/UnsafeBufferUsageFormat.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangScalableStaticAnalysisCore

  DEPENDS
  )
