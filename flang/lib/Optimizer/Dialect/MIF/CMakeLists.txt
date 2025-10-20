add_flang_library(MIFDialect
  MIFDialect.cpp
  MIFOps.cpp

  DEPENDS
  MLIRIR
  MIFOpsIncGen

  LINK_LIBS
  FIRDialect
  FIRDialectSupport
  FIRSupport
  MLIRIR
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
