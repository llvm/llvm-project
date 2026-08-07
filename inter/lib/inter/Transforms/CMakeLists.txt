add_mlir_library(InterTransforms
  InterSelect.cpp
  InterNormalizeCf.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterAnalysis
  InterXeMachine
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
