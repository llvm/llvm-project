# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestMathToVCIX
  TestMathToVCIXConversion.cpp

  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIRTestMathToVCIX PUBLIC
  AIIRArithDialect
  AIIRFuncDialect
  AIIRMathDialect
  AIIRVCIXDialect
  AIIRVectorDialect
  AIIRPass
  AIIRTransforms
)
