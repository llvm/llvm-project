# Exclude tests from libAIIR.so
add_aiir_library(AIIRAffineTransformsTestPasses
  TestAffineDataCopy.cpp
  TestAffineLoopUnswitching.cpp
  TestAffineLoopParametricTiling.cpp
  TestAccessAnalysis.cpp
  TestDecomposeAffineOps.cpp
  TestReifyValueBounds.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestLoopPermutation.cpp
  TestVectorizationUtils.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRAffineTransformsTestPasses PUBLIC
  AIIRArithTransforms
  AIIRAffineAnalysis
  AIIRAffineTransforms
  AIIRAffineUtils
  AIIRIR
  AIIRPass
  AIIRSupport
  AIIRMemRefDialect
  AIIRTensorDialect
  AIIRVectorUtils
  )

target_include_directories(AIIRAffineTransformsTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )

