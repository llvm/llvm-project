add_aiir_dialect_library(AIIRXeGPUTransformOps
  XeGPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/aiir/Dialect/XeGPU/TransformOps/

  DEPENDS
  AIIRXeGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRXeGPUDialect
  AIIRXeGPUTransforms
  AIIRIR
  AIIRTransformDialect
  AIIRFuncDialect
  AIIRSCFDialect
)
