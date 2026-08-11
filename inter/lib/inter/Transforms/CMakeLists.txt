add_mlir_library(InterTransforms
  InterSelect.cpp
  InterNormalizeCf.cpp
  InterConvertCalls.cpp
  InterConvertMemory.cpp
  InterInsertSync.cpp
  InterPrepareRegAlloc.cpp
  InterMachineSchedule.cpp
  InterResourceInfo.cpp
  MachineScheduler.cpp
  Xe2ScheduleModel.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterAnalysis
  InterXW
  InterXeMachine
  MLIRAnalysis
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
