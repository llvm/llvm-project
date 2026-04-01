add_aiir_dialect_library(AIIRFuncTransforms
  DuplicateFunctionElimination.cpp
  FuncConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/Transforms

  DEPENDS
  AIIRFuncTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRTransforms
  )
