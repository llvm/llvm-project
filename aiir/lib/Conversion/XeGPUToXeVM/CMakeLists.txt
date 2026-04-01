add_aiir_conversion_library(AIIRXeGPUToXeVM
  XeGPUToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/XeGPUToXeVM

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
  AIIRVectorDialect
  AIIRArithDialect
  AIIRIndexDialect
  AIIRSCFDialect
  AIIRXeGPUDialect
  AIIRXeGPUUtils
  AIIRPass
  AIIRTransforms
  AIIRSCFTransforms
)
