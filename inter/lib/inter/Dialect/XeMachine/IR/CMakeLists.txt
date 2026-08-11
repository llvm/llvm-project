add_mlir_dialect_library(InterXeMachine
  XeMachineDialect.cpp
  XeMachineOps.cpp
  XeMachineRegAlloc.cpp

  DEPENDS
  InterXeMachineIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRTransformDialect
)
