add_flang_library(FIRCodeGenDialect
  CGOps.cpp

  DEPENDS
  CGOpsIncGen

  LINK_LIBS
  FIRDialect

  MLIR_LIBS
  MLIRIR

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  )
