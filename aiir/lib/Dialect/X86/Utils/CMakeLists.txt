add_aiir_dialect_library(AIIRX86Utils
  X86Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/X86/Utils

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRVectorDialect
  )
