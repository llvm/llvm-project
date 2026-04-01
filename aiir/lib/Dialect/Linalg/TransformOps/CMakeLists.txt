add_aiir_dialect_library(AIIRLinalgTransformOps
  DialectExtension.cpp
  GPUHeuristics.cpp
  LinalgMatchOps.cpp
  LinalgTransformOps.cpp
  Syntax.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Linalg/TransformOps

  DEPENDS
  AIIRLinalgMatchOpsIncGen
  AIIRLinalgTransformOpsIncGen
  AIIRLinalgTransformEnumsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRIndexDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRParser
  AIIRSCFDialect
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  AIIRTransformDialectUtils
  AIIRVectorTransforms
  )
