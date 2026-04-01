add_aiir_unittest(AIIRInterfacesTests
  ControlFlowInterfacesTest.cpp
  DataLayoutInterfacesTest.cpp
  InferIntRangeInterfaceTest.cpp
  MemorySlotUtilsTest.cpp
  SideEffectInterfacesTest.cpp
  InferTypeOpInterfaceTest.cpp

  DEPENDS
  AIIRTestInterfaceIncGen
)
target_include_directories(AIIRInterfacesTests PRIVATE "${AIIR_BINARY_DIR}/test/lib/Dialect/Test")

aiir_target_link_libraries(AIIRInterfacesTests
  PRIVATE
  AIIRArithDialect
  AIIRControlFlowInterfaces
  AIIRDataLayoutInterfaces
  AIIRDLTIDialect
  AIIRFuncDialect
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRMemorySlotUtils
  AIIRParser
  AIIRSideEffectInterfaces
)
target_link_libraries(AIIRInterfacesTests PRIVATE AIIRTestDialect)
