add_aiir_unittest(AIIRLLVMIRTests
  LLVMTypeTest.cpp
)
aiir_target_link_libraries(AIIRLLVMIRTests
  PRIVATE
  AIIRLLVMDialect
  )
