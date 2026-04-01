add_aiir_conversion_library(AIIRNVVMToLLVM
  NVVMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/NVVMToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRNVVMDialect
  AIIRNVGPUDialect
  AIIRPass
  AIIRTransforms
  )
