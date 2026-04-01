add_aiir_dialect_library(AIIRVectorDialect
  VectorOps.cpp
  ValueBoundsOpInterfaceImpl.cpp
  ScalableValueBoundsConstraintSet.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Vector/IR

  DEPENDS
  AIIRMaskableOpInterfaceIncGen
  AIIRMaskingOpInterfaceIncGen
  AIIRVectorOpsIncGen
  AIIRVectorAttributesIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRControlFlowInterfaces
  AIIRDataLayoutInterfaces
  AIIRDestinationStyleOpInterface
  AIIRDialectUtils
  AIIRIndexingMapOpInterface
  AIIRIR
  AIIRMaskableOpInterface
  AIIRMaskingOpInterface
  AIIRMemRefDialect
  AIIRSideEffectInterfaces
  AIIRTensorDialect
  AIIRUBDialect
  AIIRValueBoundsOpInterface
  AIIRVectorInterfaces
  )
