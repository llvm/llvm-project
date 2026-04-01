add_aiir_dialect_library(AIIRTensorTransformOps
  TensorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor/TransformOps

  DEPENDS
  AIIRTensorTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRIR
  AIIRSCFDialect
  AIIRTensorDialect
  AIIRTensorTransforms
  AIIRTensorUtils
  AIIRTransformDialect
)
