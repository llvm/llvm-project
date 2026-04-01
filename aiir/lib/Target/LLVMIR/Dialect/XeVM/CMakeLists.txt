set(LLVM_OPTIONAL_SOURCES
  XeVMToLLVMIRTranslation.cpp
)

add_aiir_translation_library(AIIRXeVMToLLVMIRTranslation
  XeVMToLLVMIRTranslation.cpp

  DEPENDS
  AIIRXeVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRDialectUtils
  AIIRIR
  AIIRLLVMDialect
  AIIRXeVMDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
)
