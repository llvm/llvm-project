get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  FunctionFiltering.cpp
  GenericLoopConversion.cpp
  MapsForPrivatizedSymbols.cpp
  MapInfoFinalization.cpp
  MarkDeclareTarget.cpp
  LowerWorkshare.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  FlangOpenMPPassesIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranCommon
  HLFIRDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIROpenMPDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ${dialect_libs}
)
