add_aiir_dialect_library(AIIRArithUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Arith/Utils

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRAsmParser
  AIIRComplexDialect
  AIIRDialect
  AIIRDialectUtils
  AIIRIR
  )
