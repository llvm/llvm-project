add_aiir_conversion_library(AIIROpenACCToSCF
  OpenACCToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/OpenACCToSCF

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIR
  AIIROpenACCDialect
  AIIRSCFDialect
  AIIRTransforms
  )
