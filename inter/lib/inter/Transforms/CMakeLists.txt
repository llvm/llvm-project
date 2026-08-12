add_mlir_library(InterTransforms
  InterSelect.cpp
  InterImportLLVM.cpp
  InterConvertLLVMToXW.cpp
  InterRefineDistribution.cpp
  InterInferMemoryTokens.cpp
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
  MLIRDataLayoutInterfaces
  MLIRDialectUtils
  MLIRSCFTransforms
  MLIRTransformUtils
  MLIRUBDialect
  MLIRIR
  MLIRPass
)
