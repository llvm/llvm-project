add_aiir_conversion_library(AIIRMathToLLVM
  MathToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAttrToLLVMConversion
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMathDialect
  AIIRPass
  AIIRTransforms
  )
