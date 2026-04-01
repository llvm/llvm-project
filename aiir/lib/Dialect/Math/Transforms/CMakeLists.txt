add_aiir_dialect_library(AIIRMathTransforms
  AlgebraicSimplification.cpp
  ExpandOps.cpp
  ExtendToSupportedTypes.cpp
  PolynomialApproximation.cpp
  SincosFusion.cpp
  UpliftToFMA.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Math/Transforms

  DEPENDS
  AIIRMathTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRComplexDialect
  AIIRDialectUtils
  AIIRIR
  AIIRMathDialect
  AIIRSCFDialect
  AIIRPass
  AIIRTransforms
  AIIRX86Dialect
  AIIRVectorDialect
  )
