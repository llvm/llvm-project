# Exclude tests from libAIIR.so
add_aiir_library(AIIRMathTestPasses
  TestAlgebraicSimplification.cpp
  TestPolynomialApproximation.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRMathTestPasses PUBLIC
  AIIRMathTransforms
  AIIRPass
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRX86Dialect
  )
