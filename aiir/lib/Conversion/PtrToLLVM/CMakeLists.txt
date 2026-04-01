add_aiir_conversion_library(AIIRPtrToLLVM
  PtrToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/PtrToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRPtrDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  )
