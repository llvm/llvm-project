add_aiir_conversion_library(AIIRGPUToLLVMSPV
  GPUToLLVMSPV.cpp

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRGPUDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRSPIRVAttrToLLVMConversion
)
