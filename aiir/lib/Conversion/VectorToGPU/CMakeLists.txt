add_aiir_conversion_library(AIIRVectorToGPU
  VectorToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToGPU

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRGPUDialect
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRNVGPUDialect
  AIIRNVGPUUtils
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorTransforms
  AIIRVectorUtils
  )
