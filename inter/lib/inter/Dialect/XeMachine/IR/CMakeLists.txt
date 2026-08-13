add_mlir_dialect_library(InterXeMachine
  Xe2Timing.cpp
  XeMachineAliasAnalysis.cpp
  XeMachineDialect.cpp
  XeMachineOps.cpp
  XeMachineRegAlloc.cpp
  XeMachineRegAllocPreparation.cpp
  XeMachineRegionFlow.cpp

  DEPENDS
  InterXeMachineIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRLoopLikeInterface
  MLIRFuncDialect
  MLIRTransformDialect
)
