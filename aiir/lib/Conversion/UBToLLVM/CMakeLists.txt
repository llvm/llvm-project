add_aiir_conversion_library(AIIRUBToLLVM
  UBToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/UBToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRUBDialect
  )
