get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  OMPFunctionFiltering.cpp
  OMPMapInfoFinalization.cpp
  OMPMarkDeclareTarget.cpp

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
)
