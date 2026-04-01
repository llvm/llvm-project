# Exclude tests from libAIIR.so
add_aiir_library(AIIRSCFTestPasses
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestParallelLoopUnrolling.cpp
  TestSCFUtils.cpp
  TestSCFWrapInZeroTripCheck.cpp
  TestUpliftWhileToFor.cpp
  TestWhileOpBuilder.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRSCFTestPasses PUBLIC
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRSCFUtils
  AIIRTransformUtils
  )
