add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCDevicePtrToCUFKernel.cpp
  ACCEmitNYIFlang.cpp
  ACCInitializeFIRAnalyses.cpp
  ACCPipeline.cpp
  ACCOptimizeFirstprivateMap.cpp
  ACCRecipeBufferization.cpp
  ACCUseDeviceCanonicalizer.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  CUFDialect
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
