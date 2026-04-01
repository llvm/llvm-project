# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestConvertToSPIRV
  TestSPIRVFuncSignatureConversion.cpp
  TestSPIRVVectorUnrolling.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRTestConvertToSPIRV PUBLIC
  AIIRArithDialect
  AIIRFuncDialect
  AIIRPass
  AIIRSPIRVConversion
  AIIRSPIRVDialect
  AIIRTransformUtils
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorTransforms
  )
