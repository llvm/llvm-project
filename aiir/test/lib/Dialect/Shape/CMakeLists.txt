# Exclude tests from libAIIR.so
add_aiir_library(AIIRShapeTestPasses
  TestShapeFunctions.cpp
  TestShapeMappingAnalysis.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Shape
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR
  )
aiir_target_link_libraries(AIIRShapeTestPasses PUBLIC
  AIIRIR
  AIIRPass
  AIIRShapeOpsTransforms
  AIIRShapeDialect
  AIIRSupport
  )
