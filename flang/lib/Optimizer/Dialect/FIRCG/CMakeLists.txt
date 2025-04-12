add_flang_library(FIRCodeGenDialect
  CGOps.cpp

  DEPENDS
  CGOpsIncGen

  LINK_LIBS
  FIRDialect
  MLIRIR

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  )
