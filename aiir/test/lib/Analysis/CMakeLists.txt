# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestAnalysis
  TestAliasAnalysis.cpp
  TestCallGraph.cpp
  TestDataFlowFramework.cpp
  TestLiveness.cpp
  TestCFGLoopInfo.cpp
  TestMatchReduction.cpp
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestSlice.cpp
  TestTopologicalSort.cpp

  DataFlow/TestDeadCodeAnalysis.cpp
  DataFlow/TestDenseBackwardDataFlowAnalysis.cpp
  DataFlow/TestDenseForwardDataFlowAnalysis.cpp
  DataFlow/TestLivenessAnalysis.cpp
  DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
  DataFlow/TestStridedMetadataRangeAnalysis.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestAnalysis PUBLIC
  AIIRAffineDialect
  AIIRAnalysis
  AIIRFunctionInterfaces
  AIIRMemRefDialect
  AIIRPass
  )

target_include_directories(AIIRTestAnalysis
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
