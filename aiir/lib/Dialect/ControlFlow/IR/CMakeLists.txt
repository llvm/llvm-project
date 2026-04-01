add_aiir_dialect_library(AIIRControlFlowDialect
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ControlFlow/IR

  DEPENDS
  AIIRControlFlowOpsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRControlFlowInterfaces
  AIIRIR
  AIIRSideEffectInterfaces
  AIIRUBDialect
  )
