get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(HLFIRTransforms
  BufferizeHLFIR.cpp
  ConvertToFIR.cpp
  InlineElementals.cpp
  InlineHLFIRAssign.cpp
  LowerHLFIRIntrinsics.cpp
  LowerHLFIROrderedAssignments.cpp
  ScheduleOrderedAssignments.cpp
  SimplifyHLFIRIntrinsics.cpp
  OptimizedBufferization.cpp

  DEPENDS
  CUFAttrsIncGen
  FIRDialect
  HLFIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  CUFAttrs
  FIRAnalysis
  FIRDialect
  FIRBuilder
  FIRDialectSupport
  FIRSupport
  FIRTransforms
  FlangOpenMPTransforms
  HLFIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_LIBS
  MLIRIR
  ${dialect_libs}
)
