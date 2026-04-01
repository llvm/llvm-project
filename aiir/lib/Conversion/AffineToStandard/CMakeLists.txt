add_aiir_conversion_library(AIIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/AffineToStandard

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineTransforms
  AIIRAffineUtils
  AIIRArithDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTransforms
  AIIRVectorDialect
  )
