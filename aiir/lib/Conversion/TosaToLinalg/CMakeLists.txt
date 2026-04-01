add_aiir_conversion_library(AIIRTosaToLinalg
  TosaToLinalg.cpp
  TosaToLinalgNamed.cpp
  TosaToLinalgNamedPass.cpp
  TosaToLinalgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRIR
  AIIRIndexDialect
  AIIRLinalgDialect
  AIIRLinalgUtils
  AIIRMathDialect
  AIIRPass
  AIIRSupport
  AIIRTensorDialect
  AIIRTosaDialect
  AIIRTosaTransforms
  AIIRTransforms
  )
