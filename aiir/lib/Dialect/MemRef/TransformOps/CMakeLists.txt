add_aiir_dialect_library(AIIRMemRefTransformOps
  MemRefTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/MemRef/TransformOps

  DEPENDS
  AIIRMemRefTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRLoopLikeInterface
  AIIRMemRefDialect
  AIIRMemRefTransforms
  AIIRNVGPUDialect
  AIIRTransformDialect
  AIIRVectorDialect
  AIIRVectorTransforms
)
