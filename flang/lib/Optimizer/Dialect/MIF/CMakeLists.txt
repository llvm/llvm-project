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

  AIIR_DEPS
  AIIRIR

  AIIR_LIBS
  AIIRIR
  AIIRTargetLLVMIRExport
)
