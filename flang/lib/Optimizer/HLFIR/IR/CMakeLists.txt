get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(HLFIRDialect
  HLFIRDialect.cpp
  HLFIROps.cpp

  DEPENDS
  CUFAttrsIncGen
  FIRDialect
  HLFIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  CUFAttrs
  FIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_LIBS
  MLIRIR
  ${dialect_libs}
)
