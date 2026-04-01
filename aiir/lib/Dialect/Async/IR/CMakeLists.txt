add_aiir_dialect_library(AIIRAsyncDialect
  Async.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Async

  DEPENDS
  AIIRAsyncOpsIncGen

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRDialect
  AIIRInferTypeOpInterface
  AIIRIR
  )
