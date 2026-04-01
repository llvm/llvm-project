add_aiir_conversion_library(AIIRTosaToTensor
  TosaToTensor.cpp
  TosaToTensorPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRTensorDialect
  AIIRTensorUtils
  AIIRIR
  AIIRPass
  AIIRTosaDialect
  AIIRTosaTransforms
  AIIRSupport
  )
