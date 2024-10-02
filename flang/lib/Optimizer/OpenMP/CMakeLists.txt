get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  FunctionFiltering.cpp
  MapInfoFinalization.cpp
  MarkDeclareTarget.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  FlangOpenMPPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranCommon
  MLIRFuncDialect
  MLIROpenMPDialect
  HLFIRDialect
  MLIRIR
  MLIRPass
)
