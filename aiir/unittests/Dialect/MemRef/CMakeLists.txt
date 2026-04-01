add_aiir_unittest(AIIRMemRefTests
  InferShapeTest.cpp
)
aiir_target_link_libraries(AIIRMemRefTests
  PRIVATE
  AIIRMemRefDialect
  )
