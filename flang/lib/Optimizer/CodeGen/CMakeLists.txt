get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRCodeGen
  CodeGen.cpp

  DEPENDS
  FIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
)
