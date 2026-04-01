set(LLVM_OPTIONAL_SOURCES
  LLVMIRToLLVMTranslation.cpp
  LLVMToLLVMIRTranslation.cpp
  )

add_aiir_translation_library(AIIRLLVMIRToLLVMTranslation
  LLVMIRToLLVMTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRImport
  )

add_aiir_translation_library(AIIRLLVMToLLVMIRTranslation
  LLVMToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  )
