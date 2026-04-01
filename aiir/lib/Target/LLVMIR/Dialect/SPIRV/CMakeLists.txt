add_aiir_translation_library(AIIRSPIRVToLLVMIRTranslation
  SPIRVToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSPIRVDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
