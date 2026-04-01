add_aiir_unittest(AIIRDialectUtilsTests
  StructuredOpsUtilsTest.cpp
  ReshapeOpsUtilsTest.cpp
  IndexingUtilsTest.cpp
)
aiir_target_link_libraries(AIIRDialectUtilsTests
  PRIVATE
  AIIRDialectUtils)
