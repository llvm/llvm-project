# Exclude tests from libAIIR.so
add_aiir_library(AIIRVectorTestPasses
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRVectorTestPasses PUBLIC
  AIIRAffineDialect
  AIIRAnalysis
  AIIRArithDialect
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTensorDialect
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorToSCF
  AIIRX86Dialect
  )
