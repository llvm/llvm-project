add_aiir_unittest(AIIRIndexOpsTests
  IndexOpsFoldersTest.cpp
)
aiir_target_link_libraries(AIIRIndexOpsTests
  PRIVATE
  AIIRIndexDialect
)
