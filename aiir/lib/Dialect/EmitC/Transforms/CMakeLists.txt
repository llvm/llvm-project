add_aiir_dialect_library(AIIREmitCTransforms
  Transforms.cpp
  FormExpressions.cpp
  TypeConversions.cpp
  WrapFuncInClass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/EmitC/Transforms

  DEPENDS
  AIIREmitCTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRPass
  AIIREmitCDialect
  AIIRTransforms
)
