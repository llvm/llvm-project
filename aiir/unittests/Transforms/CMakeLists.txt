add_aiir_unittest(AIIRTransformsTests
  Canonicalizer.cpp
  DialectConversion.cpp
)
aiir_target_link_libraries(AIIRTransformsTests
  PRIVATE
  AIIRParser
  AIIRTransforms)
