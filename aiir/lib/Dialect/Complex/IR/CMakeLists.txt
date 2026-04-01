add_aiir_dialect_library(AIIRComplexDialect
  ComplexOps.cpp
  ComplexDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Complex

  DEPENDS
  AIIRComplexOpsIncGen
  AIIRComplexAttributesIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialect
  AIIRInferTypeOpInterface
  AIIRIR
  )
