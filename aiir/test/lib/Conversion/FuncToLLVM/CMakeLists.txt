# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestFuncToLLVM
  TestConvertCallOp.cpp
  TestConvertFuncOp.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRTestFuncToLLVM PUBLIC
  AIIRFuncToLLVM
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRLLVMIRTransforms
  AIIRPass
  )

target_include_directories(AIIRTestFuncToLLVM
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
