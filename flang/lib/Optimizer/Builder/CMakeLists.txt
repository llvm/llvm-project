get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRBuilder
  BoxValue.cpp
  Character.cpp
  Complex.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  HLFIRTools.cpp
  IntrinsicCall.cpp
  LowLevelIntrinsics.cpp
  MutableBox.cpp
  Runtime/Allocatable.cpp
  Runtime/ArrayConstructor.cpp
  Runtime/Assign.cpp
  Runtime/Character.cpp
  Runtime/Command.cpp
  Runtime/Derived.cpp
  Runtime/EnvironmentDefaults.cpp
  Runtime/Inquiry.cpp
  Runtime/Intrinsics.cpp
  Runtime/Numeric.cpp
  Runtime/Ragged.cpp
  Runtime/Reduction.cpp
  Runtime/Stop.cpp
  Runtime/Transformational.cpp

  DEPENDS
  FIRDialect
  HLFIRDialect
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect
  ${dialect_libs}
)
