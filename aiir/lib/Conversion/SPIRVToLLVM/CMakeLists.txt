add_aiir_conversion_library(AIIRSPIRVToLLVM
  ConvertLaunchFuncToLLVMCalls.cpp
  SPIRVToLLVM.cpp
  SPIRVToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SPIRVToLLVM

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_LIBS PUBLIC
  AIIRArithToLLVM
  AIIRFuncToLLVM
  AIIRGPUDialect
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefToLLVM
  AIIRSPIRVAttrToLLVMConversion
  AIIRSPIRVDialect
  AIIRSPIRVUtils
  AIIRTransforms
  )
