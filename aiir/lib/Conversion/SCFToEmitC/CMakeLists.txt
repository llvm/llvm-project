add_aiir_conversion_library(AIIRSCFToEmitC
  SCFToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SCFToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIREmitCDialect
  AIIREmitCTransforms
  AIIRSCFDialect
  AIIRTransforms
  )
