add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCDevicePtrToCUFKernel.cpp
  ACCEmitNYIFlang.cpp
  ACCEraseUnusedKernelAllocations.cpp
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
  MLIRMemRefDialect
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRViewLikeInterface
  MLIRTransformUtils
)
