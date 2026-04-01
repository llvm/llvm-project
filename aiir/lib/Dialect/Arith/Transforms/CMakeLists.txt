add_aiir_dialect_library(AIIRArithTransforms
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  EmulateUnsupportedFloats.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExpandOps.cpp
  IntRangeOptimizations.cpp
  ReifyValueBounds.cpp
  ShardingInterfaceImpl.cpp
  UnsignedWhenEquivalent.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Arith/Transforms

  DEPENDS
  AIIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRFuncDialect
  AIIRFuncTransforms
  AIIRInferIntRangeInterface
  AIIRIR
  AIIRLoopLikeInterface
  AIIRMemRefDialect
  AIIRShardDialect
  AIIRPass
  AIIRShardingInterface
  AIIRTensorDialect
  AIIRTransforms
  AIIRTransformUtils
  AIIRValueBoundsOpInterface
  AIIRVectorDialect
  )
