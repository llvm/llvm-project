add_aiir_conversion_library(AIIRTensorToSPIRV
  TensorToSPIRV.cpp
  TensorToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithToSPIRV
  AIIRFuncToSPIRV
  AIIRIR
  AIIRPass
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSupport
  AIIRTransformUtils
  AIIRTensorDialect
  )
