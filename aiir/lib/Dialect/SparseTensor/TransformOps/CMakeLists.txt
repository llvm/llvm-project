add_aiir_dialect_library(AIIRSparseTensorTransformOps
  SparseTensorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SparseTensor/TransformOps

  DEPENDS
  AIIRSparseTensorTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSparseTensorDialect
  AIIRLinalgTransformOps
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  AIIRTransformDialectUtils
  )
