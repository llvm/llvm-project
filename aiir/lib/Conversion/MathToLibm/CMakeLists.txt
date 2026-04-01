add_aiir_conversion_library(AIIRMathToLibm
  MathToLibm.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToLibm

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRMathDialect
  AIIRPass
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorUtils
  )
