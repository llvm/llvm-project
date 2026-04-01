# Exclude tests from libAIIR.so
add_aiir_library(AIIRSPIRVTestPasses
  TestAvailability.cpp
  TestEntryPointAbi.cpp
  TestModuleCombiner.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR
  )
aiir_target_link_libraries(AIIRSPIRVTestPasses PUBLIC
  AIIRGPUDialect
  AIIRIR
  AIIRPass
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSPIRVModuleCombiner
  AIIRSupport
  )
