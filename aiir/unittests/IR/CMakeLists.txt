add_aiir_unittest(AIIRIRTests
  AdaptorTest.cpp
  AffineExprTest.cpp
  AffineMapTest.cpp
  AttributeTest.cpp
  AttrTypeReplacerTest.cpp
  BuiltinDialectVersionTest.cpp
  Diagnostic.cpp
  DialectTest.cpp
  DistinctAttributeAllocatorTest.cpp
  InterfaceTest.cpp
  IRMapping.cpp
  InterfaceAttachmentTest.cpp
  LocationTest.cpp
  MemrefLayoutTest.cpp
  OperationSupportTest.cpp
  PatternMatchTest.cpp
  RemarkTest.cpp  
  ShapedTypeTest.cpp
  SymbolTableTest.cpp
  TypeTest.cpp
  TypeAttrNamesTest.cpp
  OpPropertiesTest.cpp
  ValueTest.cpp
  VerifierTest.cpp
  BlobManagerTest.cpp

  DEPENDS
  AIIRTestInterfaceIncGen
)
target_include_directories(AIIRIRTests PRIVATE "${AIIR_BINARY_DIR}/test/lib/Dialect/Test")
aiir_target_link_libraries(AIIRIRTests PRIVATE AIIRIR)
target_link_libraries(AIIRIRTests PRIVATE AIIRBytecodeWriter)
target_link_libraries(AIIRIRTests PRIVATE AIIRTestDialect)
target_link_libraries(AIIRIRTests PRIVATE AIIRRemarkStreamer)
