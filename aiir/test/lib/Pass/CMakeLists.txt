get_property(conversion_libs GLOBAL PROPERTY AIIR_CONVERSION_LIBS)
# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestPass
  TestConvertToSPIRVPass.cpp
  TestDynamicPipeline.cpp
  TestPassManager.cpp
  TestRemarksPass.cpp
  TestSPIRVCPURunnerPipeline.cpp
  TestVulkanRunnerPipeline.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Pass

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestPass PUBLIC
  ${conversion_libs}
  AIIRIR
  AIIRSPIRVTransforms
  AIIRPass
  )

target_include_directories(AIIRTestPass
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
