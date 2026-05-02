add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCInitializeFIRAnalyses.cpp
  ACCOptimizeFirstprivateMap.cpp
  ACCRecipeBufferization.cpp
  ACCUseDeviceCanonicalizer.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIROpenACCAnalysis
  FIROpenACCSupport
  HLFIRDialect

  MLIR_LIBS
  MLIRIR
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransformUtils
)
