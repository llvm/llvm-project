set(LLVM_OPTIONAL_SOURCES
  LLVMIRToNVVMTranslation.cpp
  NVVMToLLVMIRTranslation.cpp
  )

add_aiir_translation_library(AIIRLLVMIRToNVVMTranslation
  LLVMIRToNVVMTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRNVVMDialect
  AIIRSupport
  AIIRTargetLLVMIRImport
  )

add_aiir_translation_library(AIIRNVVMToLLVMIRTranslation
  NVVMToLLVMIRTranslation.cpp

  DEPENDS
  AIIRNVVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRDialectUtils
  AIIRIR
  AIIRLLVMDialect
  AIIRNVVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
