add_aiir_conversion_library(AIIRMPIToLLVM
  MPIToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MPIToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRControlFlowDialect
  AIIRDLTIDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMPIDialect
)
