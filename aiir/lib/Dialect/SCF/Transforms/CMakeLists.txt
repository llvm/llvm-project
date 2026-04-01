add_aiir_dialect_library(AIIRSCFTransforms
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  ForallToFor.cpp
  ForallToParallel.cpp
  ForToWhile.cpp
  LoopCanonicalization.cpp
  LoopPipelining.cpp
  LoopRangeFolding.cpp
  LoopSpecialization.cpp
  ParallelForToNestedFors.cpp
  ParallelLoopCollapsing.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
  RotateWhileLoop.cpp
  StructuralTypeConversions.cpp
  TileUsingInterface.cpp
  WrapInZeroTripCheck.cpp
  UpliftWhileToFor.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SCF

  DEPENDS
  AIIRSCFPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRDestinationStyleOpInterface
  AIIRDialectUtils
  AIIRIndexDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRSCFUtils
  AIIRSideEffectInterfaces
  AIIRSupport
  AIIRTensorTransforms
  AIIRTransforms
  AIIRTransformUtils
)
