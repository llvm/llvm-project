get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRBuilder
  BoxValue.cpp
  Character.cpp
  Complex.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  MutableBox.cpp
  Runtime/Assign.cpp
  Runtime/Character.cpp
  Runtime/Derived.cpp
  Runtime/Numeric.cpp
  Runtime/Ragged.cpp
  Runtime/Reduction.cpp
  Runtime/Transformational.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
  ${dialect_libs}
)
