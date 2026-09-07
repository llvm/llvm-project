add_flang_library(flangPasses
  CommandLineOpts.cpp
  Pipelines.cpp

  LINK_LIBS
  FIRCodeGen
  FIROpenACCTransforms
  FIRTransforms
  FlangOpenMPTransforms
  FortranSupport
  HLFIRTransforms

  LINK_COMPONENTS
  Passes

  MLIR_LIBS
  ${FLANG_MLIR_DIALECT_LIBS}
  ${FLANG_MLIR_EXTENSION_LIBS}
  MLIRComplexToStandard
  MLIRLLVMIRTransforms
  MLIRMathToFuncs
  MLIRMathToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
)
