add_aiir_translation_library(AIIRArmSVEToLLVMIRTranslation
  ArmSVEToLLVMIRTranslation.cpp

  DEPENDS
  AIIRArmSVEConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRArmSVEDialect
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
