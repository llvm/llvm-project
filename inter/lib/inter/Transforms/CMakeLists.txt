add_mlir_library(InterTransforms
  InterSelect.cpp
  InterNormalizeCf.cpp
  InterConvertCalls.cpp
  InterConvertMemory.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterAnalysis
  InterXW
  InterXeMachine
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
