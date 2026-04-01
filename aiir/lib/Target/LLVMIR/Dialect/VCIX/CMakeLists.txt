add_aiir_translation_library(AIIRVCIXToLLVMIRTranslation
  VCIXToLLVMIRTranslation.cpp

  DEPENDS
  AIIRVCIXConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRVCIXDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
