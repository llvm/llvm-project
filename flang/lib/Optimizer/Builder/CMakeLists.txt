get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_flang_library(FIRBuilder
  BoxValue.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
  ${dialect_libs}
)
