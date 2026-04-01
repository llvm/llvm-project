add_aiir_dialect_library(AIIRDLTITransformOps
  DLTITransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/DLTI/TransformOps

  DEPENDS
  AIIRDLTITransformOpsIncGen
  AIIRDLTIDialect

  LINK_LIBS PUBLIC
  AIIRDLTIDialect
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  )
