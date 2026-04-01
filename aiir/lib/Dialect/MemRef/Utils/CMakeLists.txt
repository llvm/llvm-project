add_aiir_dialect_library(AIIRMemRefUtils
  MemRefUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/aiir/Dialect/MemRef/Utils

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRAffineDialect
  AIIRArithUtils
)
