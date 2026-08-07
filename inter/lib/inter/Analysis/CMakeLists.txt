add_mlir_library(InterAnalysis
  UniformityAnalysis.cpp

  LINK_LIBS PUBLIC
  InterXeMachine
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRAnalysis
  MLIRIR
)
