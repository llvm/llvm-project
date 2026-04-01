add_aiir_dialect_library(AIIRAffineAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine

  DEPENDS
  AIIRFuncOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAnalysis
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRDialectUtils
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  AIIRPresburger
  AIIRSCFDialect
  )
