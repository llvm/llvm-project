add_mlir_dialect_library(InterXeMachine
  Xe2Timing.cpp
  XeMachineABI.cpp
  XeMachineAliasAnalysis.cpp
  XeMachineDialect.cpp
  XeMachineOps.cpp
  XeMachineRegAlloc.cpp
  XeMachineRegAllocPreparation.cpp
  XeMachineRegionFlow.cpp
  XeMachineTarget.cpp

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
