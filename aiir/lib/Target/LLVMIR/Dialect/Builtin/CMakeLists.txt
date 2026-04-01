add_aiir_translation_library(AIIRBuiltinToLLVMIRTranslation
  BuiltinToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
