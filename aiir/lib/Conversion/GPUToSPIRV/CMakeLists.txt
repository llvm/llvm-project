add_aiir_conversion_library(AIIRGPUToSPIRV
  GPUToSPIRV.cpp
  GPUToSPIRVPass.cpp
  WmmaOpsToSPIRV.cpp

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithToSPIRV
  AIIRGPUDialect
  AIIRFuncToSPIRV
  AIIRIR
  AIIRPass
  AIIRSCFToSPIRV
  AIIRVectorToSPIRV
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSupport
  AIIRTransforms
  )
