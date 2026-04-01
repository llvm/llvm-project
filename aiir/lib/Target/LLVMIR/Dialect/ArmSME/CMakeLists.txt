add_aiir_translation_library(AIIRArmSMEToLLVMIRTranslation
  ArmSMEToLLVMIRTranslation.cpp

  DEPENDS
  AIIRArmSMEConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRArmSMEDialect
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
