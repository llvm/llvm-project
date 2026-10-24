add_mlir_library(InterAnalysis
  DistributionAnalysis.cpp
  MemoryFrontierAnalysis.cpp

  LINK_LIBS PUBLIC
  InterXW
  MLIRArithDialect
  MLIRFuncDialect
  MLIRAnalysis
  MLIRIR
  MLIRUBDialect
)
