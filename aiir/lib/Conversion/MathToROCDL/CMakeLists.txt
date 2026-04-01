add_aiir_conversion_library(AIIRMathToROCDL
  MathToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToROCDL

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAMDGPUUtils
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRMathDialect
  AIIRLLVMCommonConversion
  AIIRPass
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorUtils
  )
