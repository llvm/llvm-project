add_aiir_conversion_library(AIIRComplexToStandard
  ComplexToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ComplexToStandard

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRComplexDivisionConversion
  AIIRComplexDialect
  AIIRIR
  AIIRMathDialect
  AIIRTransforms
  )
