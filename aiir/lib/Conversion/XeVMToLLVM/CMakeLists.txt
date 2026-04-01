add_aiir_conversion_library(AIIRXeVMToLLVM
  XeVMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/XeVMToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRXeVMDialect
  AIIRPass
  AIIRTransforms
)
