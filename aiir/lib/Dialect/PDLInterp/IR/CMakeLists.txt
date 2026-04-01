add_aiir_dialect_library(AIIRPDLInterpDialect
  PDLInterp.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/PDLInterp

  DEPENDS
  AIIRPDLInterpOpsIncGen

  LINK_LIBS PUBLIC
  AIIRFunctionInterfaces
  AIIRIR
  AIIRPDLDialect
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  )
