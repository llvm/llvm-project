# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestVectorToSPIRV
  TestVectorReductionToSPIRVDotProd.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRTestVectorToSPIRV PUBLIC
  AIIRVectorToSPIRV
  AIIRArithDialect
  AIIRFuncDialect
  AIIRSPIRVDialect
  AIIRVectorDialect
  AIIRPass
  AIIRTransforms
  )
