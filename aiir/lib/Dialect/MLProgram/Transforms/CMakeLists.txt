add_aiir_dialect_library(AIIRMLProgramTransforms
  BufferizableOpInterfaceImpl.cpp
  PipelineGlobalOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/MLProgram/Transforms

  DEPENDS
  AIIRMLProgramPassIncGen

  LINK_LIBS PUBLIC
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRIR
  AIIRMLProgramDialect
  AIIRPass
)
