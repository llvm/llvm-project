set(LLVM_OPTIONAL_SOURCES
  AliasAnalysis.cpp
  CallGraph.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  CFGLoopInfo.cpp
  SliceAnalysis.cpp
  TopologicalSortUtils.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/LivenessAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  )

add_subdirectory(Presburger)

add_aiir_library(AIIRAnalysis
  AliasAnalysis.cpp
  CallGraph.cpp
  DataFlowFramework.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  CFGLoopInfo.cpp
  SliceAnalysis.cpp
  SliceWalk.cpp
  TopologicalSortUtils.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/LivenessAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  DataFlow/StridedMetadataRangeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Analysis

  DEPENDS
  aiir-headers

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRDataLayoutInterfaces
  AIIRFunctionInterfaces
  AIIRInferIntRangeInterface
  AIIRInferStridedMetadataInterface
  AIIRInferTypeOpInterface
  AIIRLoopLikeInterface
  AIIRPresburger
  AIIRSideEffectInterfaces
  AIIRViewLikeInterface
  )

