add_aiir_dialect_library(AIIREmitCDialect
  EmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/EmitC

  DEPENDS
  AIIREmitCIncGen
  AIIREmitCAttributesIncGen

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRCastInterfaces
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRIR
  AIIRSideEffectInterfaces
  )
