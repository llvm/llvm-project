add_flang_library(MIFDialect
  MIFDialect.cpp
  MIFOps.cpp

  DEPENDS
  MIFOpsIncGen

  LINK_LIBS
  FIRDialect
  FIRDialectSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  MLIRIR
  MLIRTargetLLVMIRExport
)
