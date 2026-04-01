add_aiir_dialect_library(AIIRGPUTransformOps
  GPUTransformOps.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/GPU/TransformOps
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Interfaces

  DEPENDS
  AIIRGPUTransformOpsIncGen
  AIIRDeviceMappingInterfacesIncGen
  AIIRGPUDeviceMapperEnumsGen

  LINK_LIBS PUBLIC
  AIIRGPUDialect
  AIIRGPUTransforms
  AIIRIR
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  AIIRVectorDialect
  AIIRVectorTransforms

  # ConversionPatterns
  AIIRAMDGPUToROCDL
  AIIRGPUToNVVMTransforms
  AIIRGPUToROCDLTransforms
  AIIRNVGPUToNVVM
  )
