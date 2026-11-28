add_flang_library(FIROpenACCUtils
  DescriptorScalarization.cpp

  LINK_LIBS
  FIRDialect

  MLIR_LIBS
  MLIRArithDialect
  MLIRIR
)
