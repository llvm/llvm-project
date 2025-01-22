get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRDialectSupport
  KindMapping.cpp
  FIRContext.cpp

  DEPENDS
  MLIRIR
  intrinsics_gen

  MLIR_LIBS
  ${dialect_libs}
)
