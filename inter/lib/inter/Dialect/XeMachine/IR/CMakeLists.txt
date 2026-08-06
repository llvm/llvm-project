add_mlir_dialect_library(InterXeMachine
  XeMachineDialect.cpp
  XeMachineOps.cpp

  DEPENDS
  InterXeMachineIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
)
