add_aiir_translation_library(AIIRGPUToLLVMIRTranslation
  GPUToLLVMIRTranslation.cpp
  SelectObjectAttr.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRGPUDialect
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
