add_flang_library(FIROpenACCTransforms
  ACCInitializeFIRAnalyses.cpp
  ACCRecipeBufferization.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRDialect
  FIROpenACCAnalysis
  HLFIRDialect

  MLIR_LIBS
  MLIRIR
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
)
