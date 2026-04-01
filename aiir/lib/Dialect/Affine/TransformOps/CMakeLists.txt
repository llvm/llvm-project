add_aiir_dialect_library(AIIRAffineTransformOps
  AffineTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine/TransformOps

  DEPENDS
  AIIRAffineTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineAnalysis
  AIIRAffineDialect
  AIIRFuncDialect
  AIIRIR
  AIIRAffineDialect
  AIIRAffineTransforms
  AIIRAffineUtils
  AIIRTransformDialect
  AIIRVectorDialect
)
