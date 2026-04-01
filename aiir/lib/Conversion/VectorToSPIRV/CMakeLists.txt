add_aiir_conversion_library(AIIRVectorToSPIRV
  VectorToSPIRV.cpp
  VectorToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToSPIRV

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRVectorDialect
  AIIRTransforms
  AIIRUBToSPIRV
  )
