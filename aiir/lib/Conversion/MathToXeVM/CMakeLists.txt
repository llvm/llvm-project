add_aiir_conversion_library(AIIRMathToXeVM
  MathToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToXeVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAttrToLLVMConversion
  AIIRArithDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMathDialect
  AIIRXeVMDialect
  AIIRPass
  AIIRTransforms
  )
