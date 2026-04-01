# Exclude tests from libAIIR.so
add_aiir_library(AIIRControlFlowTestPasses
  TestAssert.cpp

  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIRControlFlowTestPasses PUBLIC
  AIIRControlFlowToLLVM
  AIIRFuncDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRPass
  AIIRTransforms
)
