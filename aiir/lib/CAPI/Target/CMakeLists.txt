add_aiir_upstream_c_api_library(AIIRCAPITarget
  LLVMIR.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRToLLVMIRTranslationRegistration
  AIIRCAPIIR
  AIIRLLVMToLLVMIRTranslation
  AIIRLLVMIRToLLVMTranslation
  AIIRSupport
)

add_aiir_upstream_c_api_library(AIIRCAPIExportSMTLIB
  ExportSMTLIB.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRExportSMTLIB
)
