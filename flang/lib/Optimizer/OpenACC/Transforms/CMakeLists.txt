add_flang_library(FIROpenACCTransforms
  ACCRecipeBufferization.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  FIRDialect
  MLIROpenACCDialect
)
