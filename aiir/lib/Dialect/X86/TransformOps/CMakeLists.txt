add_aiir_dialect_library(AIIRX86TransformOps
  X86TransformOps.cpp

  DEPENDS
  AIIRX86TransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRSideEffectInterfaces
  AIIRTransformDialect
  AIIRTransformDialectUtils
  AIIRX86Dialect
  AIIRX86Transforms
  )
