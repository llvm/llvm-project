add_aiir_dialect_library(AIIRQuantTransforms
  LowerQuantOps.cpp
  NormalizeQuantTypes.cpp
  StripFuncQuantTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Quant/Transforms

  DEPENDS
  AIIRQuantTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRFuncDialect
  AIIRFuncTransforms
  AIIRIndexDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRLinalgUtils
  AIIRPass
  AIIRQuantDialect
  AIIRShapeDialect
  AIIRTensorDialect
  AIIRTransforms
  AIIRTransformUtils

  )
