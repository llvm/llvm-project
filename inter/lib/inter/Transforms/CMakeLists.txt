add_mlir_library(InterTransforms
  InterSelect.cpp
  InterNormalizeCf.cpp
  InterConvertCalls.cpp
  InterConvertMemory.cpp
  InterInsertSync.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterAnalysis
  InterXW
  InterXeMachine
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
