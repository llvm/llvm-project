add_aiir_conversion_library(AIIRComplexToROCDLLibraryCalls
  ComplexToROCDLLibraryCalls.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ComplexToROCDLLibraryCalls

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRComplexDialect
  AIIRFuncDialect
  AIIRPass
  AIIRTransformUtils
  )
