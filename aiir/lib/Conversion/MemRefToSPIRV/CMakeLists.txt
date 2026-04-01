add_aiir_conversion_library(AIIRMemRefToSPIRV
  MapMemRefStorageClassPass.cpp
  MemRefToSPIRV.cpp
  MemRefToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRFunctionInterfaces
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSupport
  AIIRTransformUtils
  )

