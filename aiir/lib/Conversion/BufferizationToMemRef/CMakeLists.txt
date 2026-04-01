add_aiir_conversion_library(AIIRBufferizationToMemRef
  BufferizationToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/BufferizationToMemRef

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRSCFDialect
  AIIRFuncDialect
  AIIRArithDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRTransforms
  )
