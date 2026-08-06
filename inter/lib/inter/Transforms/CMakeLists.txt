add_mlir_library(InterTransforms
  InterSelect.cpp

  DEPENDS
  InterPassesIncGen

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
