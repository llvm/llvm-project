get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRDialectSupport
  KindMapping.cpp
  FIRContext.cpp

  DEPENDS
  intrinsics_gen

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  ${dialect_libs}
)
