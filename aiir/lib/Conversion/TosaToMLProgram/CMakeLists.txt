add_aiir_conversion_library(AIIRTosaToMLProgram
  TosaToMLProgram.cpp
  TosaToMLProgramPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRMLProgramDialect
  AIIRPass
  AIIRTosaDialect
  AIIRTosaTransforms
  AIIRSupport
  )
