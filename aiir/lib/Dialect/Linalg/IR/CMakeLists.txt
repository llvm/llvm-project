add_aiir_dialect_library(AIIRLinalgDialect
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgDialect.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Linalg

  DEPENDS
  AIIRLinalgInterfacesIncGen
  AIIRLinalgOpsAttributesIncGen
  AIIRLinalgOpsEnumsIncGen
  AIIRLinalgOpsIncGen
  AIIRLinalgStructuredOpsIncGen
  AIIRLinalgRelayoutOpsIncGen
  AIIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRArithUtils
  AIIRBufferizationDialect
  AIIRDestinationStyleOpInterface
  AIIRDialectUtils
  AIIRFunctionInterfaces
  AIIRIndexingMapOpInterface
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRSparseTensorDialect
  AIIRSCFDialect
  AIIRMathDialect
  AIIRMemRefDialect
  AIIRTensorDialect
  AIIRValueBoundsOpInterface
  AIIRViewLikeInterface
  )
