add_aiir_dialect_library(AIIRNVGPUTransforms
  CreateAsyncGroups.cpp
  OptimizeSharedMemory.cpp
  MmaSyncTF32Transform.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/NVGPU

  DEPENDS
  AIIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRGPUDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRNVGPUDialect
  AIIRPass
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorUtils
)
