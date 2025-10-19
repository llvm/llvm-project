add_flang_library(FIROpenACCTransforms
  ACCRecipeBufferization.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  FIRDialect

  MLIR_LIBS
  MLIRIR
  MLIRPass
  MLIROpenACCDialect
)
