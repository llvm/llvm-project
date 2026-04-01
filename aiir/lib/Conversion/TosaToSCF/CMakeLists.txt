add_aiir_conversion_library(AIIRTosaToSCF
  TosaToSCF.cpp
  TosaToSCFPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSCFDialect
  AIIRPass
  AIIRTensorDialect
  AIIRTosaDialect
  AIIRTosaTransforms
  AIIRSupport
  )
