# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestIR
  TestAffineWalk.cpp
  TestBytecodeRoundtrip.cpp
  TestBuiltinAttributeInterfaces.cpp
  TestBuiltinDistinctAttributes.cpp
  TestClone.cpp
  TestDiagnostics.cpp
  TestDiagnosticsMetadata.cpp
  TestDominance.cpp
  TestFunc.cpp
  TestInterfaces.cpp
  TestMatchers.cpp
  TestLazyLoading.cpp
  TestOpaqueLoc.cpp
  TestOperationEquals.cpp
  TestPrintDefUse.cpp
  TestPrintInvalid.cpp
  TestPrintNesting.cpp
  TestSideEffects.cpp
  TestSlicing.cpp
  TestSymbolUses.cpp
  TestRegions.cpp
  TestTypes.cpp
  TestUseListOrders.cpp
  TestVisitors.cpp
  TestVisitorsGeneric.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )

aiir_target_link_libraries(AIIRTestIR PUBLIC
  AIIRPass
  AIIRBytecodeReader
  AIIRBytecodeWriter
  AIIRFunctionInterfaces
  )

target_include_directories(AIIRTestIR
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
