add_aiir_unittest(AIIRPDLToPDLInterpTests
  RootOrderingTest.cpp
)
aiir_target_link_libraries(AIIRPDLToPDLInterpTests
  PRIVATE
  AIIRArithDialect
  AIIRPDLToPDLInterp
)
