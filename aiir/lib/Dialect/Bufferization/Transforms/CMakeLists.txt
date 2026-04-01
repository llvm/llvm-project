add_aiir_dialect_library(AIIRBufferizationTransforms
  Bufferize.cpp
  BufferDeallocationSimplification.cpp
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  BufferViewFlowAnalysis.cpp
  DropEquivalentBufferResults.cpp
  EmptyTensorElimination.cpp
  EmptyTensorToAllocTensor.cpp
  FuncBufferizableOpInterfaceImpl.cpp
  LowerDeallocations.cpp
  OneShotAnalysis.cpp
  OneShotModuleBufferize.cpp
  OwnershipBasedBufferDeallocation.cpp
  TensorCopyInsertion.cpp
  OptimizeAllocationLiveness.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization

  DEPENDS
  AIIRBufferizationPassIncGen
  AIIRBufferizationEnumsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRTensorDialect
  AIIRSCFDialect
  AIIRSideEffectInterfaces
  AIIRSubsetOpInterface
  AIIRTransforms
  AIIRViewLikeInterface
  AIIRSupport
)
