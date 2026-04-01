# Exclude tests from libAIIR.so
add_aiir_library(AIIRShardTest
  TestOpLowering.cpp
  TestReshardingPartition.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRShardTest PUBLIC
  AIIRShardDialect
  AIIRShardTransforms
  AIIRPass
  AIIRRewrite
  AIIRTransformUtils
  )
