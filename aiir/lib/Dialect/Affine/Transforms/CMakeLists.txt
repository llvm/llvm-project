add_aiir_dialect_library(AIIRAffineTransforms
  AffineDataCopyGeneration.cpp
  AffineExpandIndexOps.cpp
  AffineExpandIndexOpsAsAffine.cpp
  AffineLoopInvariantCodeMotion.cpp
  AffineLoopNormalize.cpp
  AffineParallelize.cpp
  AffineScalarReplacement.cpp
  DecomposeAffineOps.cpp
  FoldMemRefAliasOps.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopTiling.cpp
  LoopUnroll.cpp
  LoopUnrollAndJam.cpp
  PipelineDataTransfer.cpp
  RaiseMemrefDialect.cpp
  ReifyValueBounds.cpp
  SuperVectorize.cpp
  SimplifyAffineWithBounds.cpp
  SimplifyAffineStructures.cpp
  SimplifyAffineMinMax.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine

  DEPENDS
  AIIRAffineOpsIncGen
  AIIRAffinePassIncGen
  AIIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRAffineUtils
  AIIRArithDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRMemRefUtils
  AIIRPass
  AIIRSCFUtils
  AIIRSideEffectInterfaces
  AIIRTensorDialect
  AIIRTransformUtils
  AIIRValueBoundsOpInterface
  AIIRVectorDialect
  AIIRVectorUtils
  )

