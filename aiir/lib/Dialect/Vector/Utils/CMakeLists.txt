add_aiir_dialect_library(AIIRVectorUtils
  VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Vector/Utils

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRArithDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRTensorDialect
  AIIRVectorDialect
  AIIRVectorInterfaces
  )
