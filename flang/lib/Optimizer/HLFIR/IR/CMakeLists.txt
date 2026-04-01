get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

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

  AIIR_DEPS
  ${dialect_libs}

  AIIR_LIBS
  AIIRIR
  ${dialect_libs}
)
