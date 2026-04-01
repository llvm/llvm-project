add_aiir_dialect_library(AIIRSCFUtils
  AffineCanonicalizationUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SCF

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRAnalysis
  AIIRArithDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRIR
  AIIRSCFDialect
  AIIRSideEffectInterfaces
  AIIRSupport
  AIIRTransforms
  )
