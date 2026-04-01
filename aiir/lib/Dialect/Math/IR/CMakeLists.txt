add_aiir_dialect_library(AIIRMathDialect
  MathOps.cpp
  MathDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Math

  DEPENDS
  AIIRMathOpsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialect
  AIIRIR
  AIIRUBDialect
  )
