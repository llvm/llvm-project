add_flang_library(HLFIRTransforms
  BufferizeHLFIR.cpp
  ConvertToFIR.cpp
  ExpressionSimplification.cpp
  InlineElementals.cpp
  InlineHLFIRAssign.cpp
  SeparateAllocatableAssign.cpp
  InlineHLFIRCopy.cpp
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
  FortranSupport
  HLFIRDialect

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  ${FLANG_MLIR_DIALECT_LIBS}

  MLIR_LIBS
  MLIRIR
  ${FLANG_MLIR_DIALECT_LIBS}
)
