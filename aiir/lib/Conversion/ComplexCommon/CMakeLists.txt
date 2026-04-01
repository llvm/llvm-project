add_aiir_conversion_library(AIIRComplexDivisionConversion
  DivisionConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRComplexDialect
  AIIRLLVMDialect
  AIIRMathDialect
  )
