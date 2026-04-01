# Exclude tests from libAIIR.so
add_aiir_library(AIIRDLTITestPasses
  TestDataLayoutQuery.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRDLTITestPasses PUBLIC
  AIIRDLTIDialect
  AIIRPass
  )

target_include_directories(AIIRDLTITestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
