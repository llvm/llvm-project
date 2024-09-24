add_flang_library(FIRPasses
  Pipelines.cpp

  DEPENDS
  FortranCommon

  LINK_LIBS
  FIRCodeGen
  FIRTransforms
  FlangOpenMPTransforms
  FortranCommon
  HLFIRTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms

  LINK_COMPONENTS
  Passes
)
