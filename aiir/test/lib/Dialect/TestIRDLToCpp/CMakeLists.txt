add_irdl_to_cpp_target(TestIRDLToCppGen test_irdl_to_cpp.irdl.aiir)

add_aiir_library(AIIRTestIRDLToCppDialect
  TestIRDLToCppDialect.cpp

  EXCLUDE_FROM_LIBAIIR

  DEPENDS
  TestIRDLToCppGen
)

aiir_target_link_libraries(AIIRTestIRDLToCppDialect PUBLIC
  AIIRIR
  AIIRPass
  AIIRSCFDialect
  AIIRTransforms
  AIIRTestDialect
)
