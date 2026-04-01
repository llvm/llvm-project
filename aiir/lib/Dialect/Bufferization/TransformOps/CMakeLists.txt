add_aiir_dialect_library(AIIRBufferizationTransformOps
  BufferizationTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization/TransformOps

  DEPENDS
  AIIRBufferizationTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRFunctionInterfaces
  AIIRLinalgDialect
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  )
