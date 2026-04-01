add_aiir_dialect_library(AIIRPDLDialect
  PDL.cpp
  PDLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/PDL

  DEPENDS
  AIIRPDLOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  )
