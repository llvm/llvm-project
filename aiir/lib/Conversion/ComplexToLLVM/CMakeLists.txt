add_aiir_conversion_library(AIIRComplexToLLVM
  ComplexToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ComplexToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAttrToLLVMConversion
  AIIRComplexDivisionConversion
  AIIRComplexDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRTransforms
  )
