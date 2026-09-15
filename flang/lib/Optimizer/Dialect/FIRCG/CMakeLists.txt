add_flang_library(FIRCodeGenDialect
  CGOps.cpp

  DEPENDS
  CGOpsIncGen
  CUFAttrsIncGen

  LINK_LIBS
  CUFAttrs
  FIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter

  MLIR_LIBS
  MLIRIR
  )
