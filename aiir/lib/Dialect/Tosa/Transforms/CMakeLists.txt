add_aiir_dialect_library(AIIRTosaTransforms
  TosaAttachTarget.cpp
  TosaArithConstantToConst.cpp
  TosaConvertIntegerTypeToSignless.cpp
  TosaDecomposeTransposeConv.cpp
  TosaDecomposeDepthwise.cpp
  TosaFolders.cpp
  TosaInferShapes.cpp
  TosaLayerwiseConstantFoldPass.cpp
  TosaMakeBroadcastable.cpp
  TosaOptionalDecompositions.cpp
  TosaReduceTransposes.cpp
  TosaTypeConverters.cpp
  TosaProfileCompliance.cpp
  TosaValidation.cpp
  TosaNarrowTypes.cpp
  TosaInputShape.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa/Transforms

  DEPENDS
  AIIRTosaPassIncGen

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRFuncTransformOps
  AIIRPass
  AIIRTosaDialect
  AIIRTransformUtils
  AIIRFuncTransforms
  )
