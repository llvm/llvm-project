# Exclude tests from libAIIR.so
add_aiir_library(AIIRFuncTestPasses
  TestDecomposeCallGraphTypes.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_LIBS PUBLIC
  AIIRTestDialect
  )
aiir_target_link_libraries(AIIRFuncTestPasses PUBLIC
  AIIRAffineDialect
  AIIRPass
  AIIRFuncTransforms
  AIIRTransformUtils
  )

target_include_directories(AIIRFuncTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
