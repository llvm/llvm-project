add_aiir_dialect_library(AIIRVectorTransformOps
  VectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Vector/TransformOps

  DEPENDS
  AIIRVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRVectorToLLVM
  AIIRVectorTransforms
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  AIIRVectorDialect
  AIIRVectorToSCF
  AIIRX86Transforms
  )
