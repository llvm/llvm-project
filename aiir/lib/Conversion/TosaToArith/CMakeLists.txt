add_aiir_conversion_library(AIIRTosaToArith
  TosaToArith.cpp
  TosaToArithPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIR
  AIIRPass
  AIIRTosaDialect
  AIIRTosaTransforms
  AIIRSupport
  )
