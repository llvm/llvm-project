add_flang_library(FIRDialectSupport
  KindMapping.cpp
  FIRContext.cpp

  DEPENDS
  intrinsics_gen

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  ${FLANG_MLIR_DIALECT_LIBS}
)
