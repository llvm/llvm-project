add_aiir_translation_library(AIIRPtrToLLVMIRTranslation
  PtrToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRPtrDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
