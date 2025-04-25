get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_flang_library(FIRBuilder
  BoxValue.cpp
  Character.cpp
  Complex.cpp
  CUFCommon.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  HLFIRTools.cpp
  IntrinsicCall.cpp
  LowLevelIntrinsics.cpp
  MutableBox.cpp
  PPCIntrinsicCall.cpp
  Runtime/Allocatable.cpp
  Runtime/ArrayConstructor.cpp
  Runtime/Assign.cpp
  Runtime/Character.cpp
  Runtime/Command.cpp
  Runtime/Derived.cpp
  Runtime/EnvironmentDefaults.cpp
  Runtime/Exceptions.cpp
  Runtime/Execute.cpp
  Runtime/Inquiry.cpp
  Runtime/Intrinsics.cpp
  Runtime/Main.cpp
  Runtime/Numeric.cpp
  Runtime/Pointer.cpp
  Runtime/Ragged.cpp
  Runtime/Reduction.cpp
  Runtime/Stop.cpp
  Runtime/Support.cpp
  Runtime/TemporaryStack.cpp
  Runtime/Transformational.cpp
  TemporaryStorage.cpp

  DEPENDS
  CUFAttrs
  CUFDialect
  FIRDialect
  HLFIRDialect
  ${dialect_libs}
  ${extension_libs}

  LINK_LIBS
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranEvaluate
  HLFIRDialect

  MLIR_LIBS
  ${dialect_libs}
  ${extension_libs}
)
