add_aiir_dialect_library(AIIRAMDGPUDialect
  AMDGPUAttrs.cpp
  AMDGPUDialect.cpp
  AMDGPUEnums.cpp
  AMDGPUOps.cpp
  AMDGPUTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/AMDGPU

  DEPENDS
  AIIRAMDGPUEnumsGen
  AIIRAMDGPUAttributesIncGen
  AIIRAMDGPUIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRROCDLDialect
  # Needed for GPU address space enum definition
  AIIRGPUDialect
  AIIRVectorDialect
  AIIRIR
  AIIRSideEffectInterfaces
  AIIRMemRefUtils
  )
