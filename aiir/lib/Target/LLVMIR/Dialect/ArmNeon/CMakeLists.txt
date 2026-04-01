add_aiir_translation_library(AIIRArmNeonToLLVMIRTranslation
  ArmNeonToLLVMIRTranslation.cpp

  DEPENDS
  AIIRArmNeonConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArmNeonDialect
  AIIRIR
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
