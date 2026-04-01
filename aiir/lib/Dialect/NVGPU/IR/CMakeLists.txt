add_aiir_dialect_library(AIIRNVGPUDialect
  NVGPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/NVGPU

  DEPENDS
  AIIRNVGPUOpsIncGen
  AIIRNVGPUEnumsIncGen
  AIIRNVGPUAttributesIncGen
  AIIRNVGPUTypesIncGen

  LINK_LIBS PUBLIC
  AIIRGPUDialect
  AIIRIR
  AIIRSideEffectInterfaces
  )
