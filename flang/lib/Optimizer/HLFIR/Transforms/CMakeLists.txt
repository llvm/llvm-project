get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# ConvertToFIR.cpp redefines AssignOpConversion and NoReassocOpConversion,
# ScheduleOrderedAssignments.cpp redefines isForallIndex, and
# SimplifyHLFIRIntrinsics.cpp redefines CmpCharOpConversion/IndexOpConversion.
set_source_files_properties(
  ConvertToFIR.cpp
  ScheduleOrderedAssignments.cpp
  SimplifyHLFIRIntrinsics.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_flang_library(HLFIRTransforms
  BufferizeHLFIR.cpp
  ConvertToFIR.cpp
  ExpressionSimplification.cpp
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
