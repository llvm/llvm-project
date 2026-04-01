add_aiir_conversion_library(AIIRTensorToLinalg
  TensorToLinalg.cpp
  TensorToLinalgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Linalg
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRPass
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRSupport
  AIIRTransformUtils
  AIIRTensorDialect
  )
