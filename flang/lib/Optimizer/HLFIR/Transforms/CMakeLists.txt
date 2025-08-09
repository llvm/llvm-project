get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(HLFIRTransforms
  BufferizeHLFIR.cpp
  ConvertToFIR.cpp
  InlineElementals.cpp
  InlineHLFIRAssign.cpp
  InlineHLFIRCopyIn.cpp
  LowerHLFIRIntrinsics.cpp
  LowerHLFIROrderedAssignments.cpp
  ScheduleOrderedAssignments.cpp
  SimplifyHLFIRIntrinsics.cpp
  OptimizedBufferization.cpp
  PropagateFortranVariableAttributes.cpp

  DEPENDS
  CUFAttrsIncGen
  FIRDialect
  HLFIROpsIncGen

  LINK_LIBS
  CUFAttrs
  FIRAnalysis
  FIRDialect
  FIRBuilder
  FIRDialectSupport
  FIRSupport
  FIRTransforms
  FlangOpenMPTransforms
  FortranEvaluate
  FortranSupport
  HLFIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  ${dialect_libs}

  MLIR_LIBS
  MLIRIR
  ${dialect_libs}
)
