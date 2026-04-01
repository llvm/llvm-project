add_aiir_dialect_library(AIIRFuncDialect
  FuncOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func

  DEPENDS
  AIIRFuncOpsIncGen

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRSideEffectInterfaces
  )
