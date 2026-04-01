add_aiir_dialect_library(AIIRTosaTestPasses
  TosaTestPasses.cpp
  TestAvailability.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa/Transforms

  DEPENDS
  AIIRTosaPassIncGen
  )
aiir_target_link_libraries(AIIRTosaTestPasses PUBLIC
  AIIRFuncDialect
  AIIRPass
  AIIRTosaDialect
  AIIRTransformUtils
  )
