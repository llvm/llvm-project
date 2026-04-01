# Exclude tests from libAIIR.so
add_aiir_library(AIIRArithTestPasses
  TestEmulateWideInt.cpp

  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIRArithTestPasses PUBLIC
  AIIRArithDialect
  AIIRArithTransforms
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRPass
  AIIRVectorDialect
)
