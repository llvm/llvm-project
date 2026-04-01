add_aiir_translation_library(AIIROpenACCToLLVMIRTranslation
  OpenACCToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIROpenACCDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  AIIRTransformUtils
  )
