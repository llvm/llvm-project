add_aiir_conversion_library(AIIRUBToSPIRV
  UBToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/UBToSPIRV

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRSPIRVConversion
  AIIRSPIRVDialect
  AIIRUBDialect
  )
