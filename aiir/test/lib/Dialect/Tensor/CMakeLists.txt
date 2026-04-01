# Exclude tests from libAIIR.so
add_aiir_library(AIIRTensorTestPasses
  TestTensorTransforms.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRTensorTestPasses PUBLIC
  AIIRArithDialect
  AIIRLinalgDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTensorDialect
  AIIRTensorTransforms
  AIIRTransformDialect
  AIIRTransforms
  )
