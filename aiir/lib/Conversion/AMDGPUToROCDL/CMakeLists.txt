add_aiir_conversion_library(AIIRAMDGPUToROCDL
  AMDGPUToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/AMDGPUToROCDL

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAMDGPUDialect
  AIIRAMDGPUUtils
  AIIRGPUToGPURuntimeTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRPass
  AIIRROCDLDialect
  AIIRTransforms
  )
