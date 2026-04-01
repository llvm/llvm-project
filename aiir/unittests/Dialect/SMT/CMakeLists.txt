add_aiir_unittest(AIIRSMTTests
  AttributeTest.cpp
  QuantifierTest.cpp
  TypeTest.cpp
)

aiir_target_link_libraries(AIIRSMTTests
  PRIVATE
  AIIRSMT
)
