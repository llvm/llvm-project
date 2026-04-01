add_aiir_conversion_library(AIIRSCFToSPIRV
  SCFToSPIRV.cpp
  SCFToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SCFToSPIRV

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithToSPIRV
  AIIRFuncToSPIRV
  AIIRIndexToSPIRV
  AIIRMemRefToSPIRV
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRIR
  AIIRPass
  AIIRSCFDialect
  AIIRSupport
  AIIRTransforms
  )
