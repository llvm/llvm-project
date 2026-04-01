add_aiir_dialect_library(AIIRTensorUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRArithUtils
  AIIRIR
  AIIRTensorDialect
  AIIRValueBoundsOpInterface
)
