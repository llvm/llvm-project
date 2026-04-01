add_aiir_dialect_library(AIIRAffineDialect
  AffineMemoryOpInterfaces.cpp
  AffineOps.cpp
  AffineValueMap.cpp
  InferIntRangeInterfaceImpls.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine

  DEPENDS
  AIIRAffineMemoryOpInterfacesIncGen
  AIIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRIR
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRLoopLikeInterface
  AIIRMemRefDialect
  AIIRShapedOpInterfaces
  AIIRSideEffectInterfaces
  AIIRUBDialect
  AIIRValueBoundsOpInterface
  )
