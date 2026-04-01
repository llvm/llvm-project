add_aiir_dialect_library(AIIRArmSVEDialect
  ArmSVEDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmSVE

  DEPENDS
  AIIRArmSVEIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRSideEffectInterfaces
  )
