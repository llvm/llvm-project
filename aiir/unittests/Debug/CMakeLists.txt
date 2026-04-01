add_aiir_unittest(AIIRDebugTests
  DebugCounterTest.cpp
  ExecutionContextTest.cpp
  FileLineColLocBreakpointManagerTest.cpp
)

aiir_target_link_libraries(AIIRDebugTests
  PRIVATE AIIRDebug)
