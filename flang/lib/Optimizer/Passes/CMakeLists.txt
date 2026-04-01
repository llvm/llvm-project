get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)

add_flang_library(flangPasses
  CommandLineOpts.cpp
  Pipelines.cpp

  LINK_LIBS
  FIRCodeGen
  FIRTransforms
  FlangOpenMPTransforms
  FortranSupport
  HLFIRTransforms

  LINK_COMPONENTS
  Passes

  AIIR_LIBS
  ${dialect_libs}
  ${extension_libs}
  AIIRPass
  AIIRReconcileUnrealizedCasts
  AIIRSCFToControlFlow
  AIIRSupport
  AIIRTransforms
)
