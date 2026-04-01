add_aiir_dialect_library(AIIRAMDGPUUtils
  Chipset.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/AMDGPU/Utils

  LINK_LIBS PUBLIC
  AIIRAMDGPUDialect
  AIIRSupport
  )
