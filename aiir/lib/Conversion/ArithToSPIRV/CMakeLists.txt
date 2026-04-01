add_aiir_conversion_library(AIIRArithToSPIRV
  ArithToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToSPIRV

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRFuncToSPIRV
  AIIRSPIRVConversion
  AIIRSPIRVDialect
  )
