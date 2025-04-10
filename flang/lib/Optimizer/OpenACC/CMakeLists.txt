get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIROpenACCSupport
  FIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect

  LINK_LIBS
  FIRBuilder
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect

  MLIR_DEPS
  MLIROpenACCDialect

  MLIR_LIBS
  MLIROpenACCDialect
)
