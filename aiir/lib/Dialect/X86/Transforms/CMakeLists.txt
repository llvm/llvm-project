add_aiir_dialect_library(AIIRX86Transforms
  AVXTranspose.cpp
  LegalizeForLLVMExport.cpp
  VectorContractToFMA.cpp
  VectorContractToPackedTypeDotProduct.cpp
  VectorContractBF16ToFMA.cpp
  SinkVectorProducerOps.cpp
  ShuffleVectorFMAOps.cpp
  VectorContractToAMXDotProduct.cpp

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRVectorUtils
  AIIRX86Dialect
  AIIRX86Utils
  )
