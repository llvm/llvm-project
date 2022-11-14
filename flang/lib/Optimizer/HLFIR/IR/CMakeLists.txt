get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(HLFIRDialect
  HLFIRDialect.cpp
  HLFIROps.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  MLIRIR
  ${dialect_libs}

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
