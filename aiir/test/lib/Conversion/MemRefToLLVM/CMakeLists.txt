# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestMemRefToLLVMWithTransforms
  TestMemRefToLLVMWithTransforms.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestMemRefToLLVMWithTransforms PUBLIC
  AIIRFuncToLLVM
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefTransforms
  AIIRPass
  )

target_include_directories(AIIRTestMemRefToLLVMWithTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
