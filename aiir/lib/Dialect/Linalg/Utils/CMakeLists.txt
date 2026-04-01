add_aiir_dialect_library(AIIRLinalgUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Linalg

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRAffineUtils
  AIIRArithDialect
  AIIRFuncDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRSCFDialect
  AIIRPass
  AIIRTensorUtils
  AIIRTransformUtils
)
