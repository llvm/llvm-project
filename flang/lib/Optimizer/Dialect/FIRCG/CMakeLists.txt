add_flang_library(FIRCodeGenDialect
  CGOps.cpp

  DEPENDS
  CGOpsIncGen

  LINK_LIBS
  FIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter

  AIIR_LIBS
  AIIRIR
  )
