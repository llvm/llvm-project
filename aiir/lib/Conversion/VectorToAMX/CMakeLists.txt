add_aiir_conversion_library(AIIRVectorToAMX
  VectorToAMX.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToAMX

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineUtils
  AIIRArithDialect
  AIIRLinalgUtils
  AIIRMemRefDialect
  AIIRSCFDialect
  AIIRTransforms
  AIIRVectorDialect
  AIIRX86Dialect
  )
