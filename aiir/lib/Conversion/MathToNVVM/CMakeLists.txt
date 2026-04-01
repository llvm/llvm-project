add_aiir_conversion_library(AIIRMathToNVVM
  MathToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToNVVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRMathDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRNVVMDialect
  AIIRPass
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorUtils
  )
