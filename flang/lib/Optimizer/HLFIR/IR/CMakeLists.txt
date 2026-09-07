add_flang_library(HLFIRDialect
  HLFIRDialect.cpp
  HLFIROps.cpp

  DEPENDS
  CUFAttrsIncGen
  FIRDialect
  HLFIROpsIncGen

  LINK_LIBS
  CUFAttrs
  FIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  ${FLANG_MLIR_DIALECT_LIBS}

  MLIR_LIBS
  MLIRIR
  ${FLANG_MLIR_DIALECT_LIBS}
)
