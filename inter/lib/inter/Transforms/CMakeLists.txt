add_mlir_library(InterTransforms
  InterSelect.cpp
  InterNormalizeCf.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
