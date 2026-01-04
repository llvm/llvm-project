get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIROpenMPSupport
  FIROpenMPAttributes.cpp
  RegisterOpenMPExtensions.cpp

  DEPENDS
  FIRBuilder
  FIRDialect

  LINK_LIBS
  FIRBuilder
  FIRDialect

  MLIR_DEPS
  MLIROpenMPDialect

  MLIR_LIBS
  MLIROpenMPDialect
)
