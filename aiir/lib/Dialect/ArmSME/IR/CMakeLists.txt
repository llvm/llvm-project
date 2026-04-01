add_aiir_dialect_library(AIIRArmSMEDialect
  ArmSME.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmSME

  DEPENDS
  AIIRArmSMEOpsIncGen
  AIIRArmSMEIntrinsicOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRSCFDialect
  AIIRSideEffectInterfaces
  AIIRVectorDialect
)
