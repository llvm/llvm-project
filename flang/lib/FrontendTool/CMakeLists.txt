add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  # This makes sure that the MLIR dependencies of flangFrontend (which are
  # transitively required here) are generated before this target is build.
  flangFrontend

  LINK_LIBS
  flangFrontend
  FIRCodeGen
  FIRTransforms
  HLFIRTransforms
  FlangOpenMPTransforms

  LINK_COMPONENTS
  Option
  Plugins
  Support

  MLIR_LIBS
  MLIRComplexToStandard
  MLIRLLVMIRTransforms
  MLIRMathToFuncs
  MLIRMathToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTransforms

  CLANG_LIBS
  clangBasic
  clangOptions
  clangDriver
)
