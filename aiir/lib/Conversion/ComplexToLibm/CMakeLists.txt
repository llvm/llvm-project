add_aiir_conversion_library(AIIRComplexToLibm
  ComplexToLibm.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ComplexToLibm

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRComplexDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRPass
  AIIRTransformUtils
  )
