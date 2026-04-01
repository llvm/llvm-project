add_aiir_conversion_library(AIIRMemRefToLLVM
  MemRefToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MemRefToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRDataLayoutInterfaces
  AIIRFuncDialect
  AIIRLLVMCommonConversion
  AIIRMemRefDialect
  AIIRMemRefUtils
  AIIRLLVMDialect
  AIIRTransforms
  )
