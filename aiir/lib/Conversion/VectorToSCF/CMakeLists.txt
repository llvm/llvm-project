add_aiir_conversion_library(AIIRVectorToSCF
  VectorToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToSCF

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRMemRefDialect
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorTransforms
  )
