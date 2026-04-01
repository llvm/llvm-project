add_aiir_conversion_library(AIIRIndexToSPIRV
  IndexToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/IndexToSPIRV

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIndexDialect
  AIIRSPIRVConversion
  AIIRSPIRVDialect
  )
