add_aiir_dialect_library(AIIRBufferizationPipelines
  BufferizationPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization

  LINK_LIBS PUBLIC
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRMemRefTransforms
  AIIRFuncDialect
  AIIRPass
  AIIRTransforms
)
