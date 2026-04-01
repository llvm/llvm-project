add_aiir_dialect_library(AIIRSCFTransformOps
  SCFTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SCF/TransformOps

  DEPENDS
  AIIRSCFTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRFuncDialect
  AIIRIR
  AIIRLoopLikeInterface
  AIIRSCFDialect
  AIIRSCFToControlFlow
  AIIRSCFTransforms
  AIIRSCFUtils
  AIIRTransformDialect
  AIIRVectorDialect
)
