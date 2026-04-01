add_aiir_conversion_library(AIIRArithToLLVM
  ArithToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAttrToLLVMConversion
  AIIRArithDialect
  AIIRArithTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  )
