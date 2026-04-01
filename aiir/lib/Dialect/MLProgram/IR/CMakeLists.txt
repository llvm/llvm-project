add_aiir_dialect_library(AIIRMLProgramDialect
  MLProgramOps.cpp
  MLProgramDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/MLProgram

  DEPENDS
  AIIRMLProgramOpsIncGen
  AIIRMLProgramAttributesIncGen
  AIIRMLProgramTypesIncGen

  LINK_LIBS PUBLIC
  AIIRDialect
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRInferTypeOpInterface
  AIIRTransforms
  AIIRIR
  )
