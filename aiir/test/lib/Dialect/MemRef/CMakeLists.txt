# Exclude tests from libAIIR.so
add_aiir_library(AIIRMemRefTestPasses
  TestComposeSubView.cpp
  TestEmulateNarrowType.cpp
  TestMultiBuffer.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRMemRefTestPasses PUBLIC
  AIIRPass
  AIIRMemRefDialect
  AIIRMemRefTransforms
  )

target_include_directories(AIIRMemRefTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
