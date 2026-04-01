add_aiir_unittest(AIIRParserTests
  ParserTest.cpp
  ResourceTest.cpp

  DEPENDS
  AIIRTestInterfaceIncGen
)
target_include_directories(AIIRParserTests PRIVATE "${AIIR_BINARY_DIR}/test/lib/Dialect/Test")

aiir_target_link_libraries(AIIRParserTests PRIVATE
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRIR
  AIIRParser
)
target_link_libraries(AIIRParserTests PRIVATE AIIRTestDialect)
