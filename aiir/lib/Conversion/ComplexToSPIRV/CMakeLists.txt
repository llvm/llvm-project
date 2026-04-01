add_aiir_conversion_library(AIIRComplexToSPIRV
  ComplexToSPIRV.cpp
  ComplexToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRComplexDialect
  AIIRIR
  AIIRPass
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSupport
  AIIRTransformUtils
  )

