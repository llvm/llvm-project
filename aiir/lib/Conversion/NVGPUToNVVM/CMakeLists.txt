add_aiir_conversion_library(AIIRNVGPUToNVVM
  NVGPUToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/NVGPUToNVVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRGPUDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRNVGPUDialect
  AIIRNVVMDialect
  AIIRArithDialect
  AIIRPass
  AIIRSCFTransforms
  AIIRTransforms
  )
