add_aiir_unittest(AIIRRewriteTests
  PatternBenefit.cpp
)
aiir_target_link_libraries(AIIRRewriteTests
  PRIVATE
  AIIRRewrite
  AIIRTransformUtils)
