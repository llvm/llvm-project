add_aiir_conversion_library(AIIRMathToFuncs
  MathToFuncs.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToFuncs

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRControlFlowDialect
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRMathDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorUtils
  )
