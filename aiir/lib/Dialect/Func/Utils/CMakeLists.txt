add_aiir_dialect_library(AIIRFuncUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/Utils

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRDialect
  AIIRDialectUtils
  AIIRIR
  )
