add_aiir_translation_library(AIIRROCDLToLLVMIRTranslation
  ROCDLToLLVMIRTranslation.cpp

  DEPENDS
  AIIRROCDLConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRROCDLDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
