# Exclude tests from libAIIR.so
add_aiir_library(AIIRBufferizationTestPasses
  TestOneShotModuleBufferize.cpp
  TestTensorCopyInsertion.cpp
  TestTensorLikeAndBufferLike.cpp

  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIRBufferizationTestPasses PUBLIC
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRIR
  AIIRPass
  AIIRTestDialect
)

target_include_directories(AIIRBufferizationTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
