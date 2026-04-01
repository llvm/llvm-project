add_aiir_dialect_library(AIIRNVGPUTransformOps
  NVGPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/NVGPU/TransformOps

  DEPENDS
  AIIRNVGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRNVGPUDialect
  AIIRNVGPUTransforms
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRTransformDialect
  AIIRTransformDialectUtils
  AIIRVectorTransforms

  # Conversions
  AIIRNVGPUToNVVM
  )
