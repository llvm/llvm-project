add_aiir_dialect_library(AIIROpenACCDialect
  OpenACC.cpp
  OpenACCCG.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenACC

  DEPENDS
  AIIROpenACCOpsIncGen
  AIIROpenACCEnumsIncGen
  AIIROpenACCAttributesIncGen
  AIIROpenACCMPOpsInterfacesIncGen
  AIIROpenACCOpsInterfacesIncGen
  AIIROpenACCTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRGPUDialect
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIROpenACCMPCommon
  )

