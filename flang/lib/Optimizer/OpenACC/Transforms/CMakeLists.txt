add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCDevicePtrToCUFKernel.cpp
  ACCEmitNYIFlang.cpp
  ACCInitializeFIRAnalyses.cpp
  ACCMapInfoPrep.cpp
  ACCOptimizeFirstprivateMap.cpp
  ACCPipeline.cpp
  ACCRecipeBufferization.cpp
  ACCUseDeviceCanonicalizer.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  CUFAttrs
  CUFDialect
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIROpenACCAnalysis
  FIROpenACCSupport
  FIRSupport
  HLFIRDialect

  MLIR_LIBS
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransformUtils
)
