add_aiir_dialect_library(AIIRSparseTensorUtils
  Merger.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRComplexDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRSparseTensorEnums
)
