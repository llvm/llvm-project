add_aiir_translation_library(AIIROpenMPToLLVMIRTranslation
  OpenMPToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIROpenMPDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  AIIRTransformUtils
  )
