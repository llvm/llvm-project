add_mlir_dialect_library(InterXeMachine
  Xe2Timing.cpp
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
