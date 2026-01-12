get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIROpenACCSupport
  FIROpenACCAttributes.cpp
  FIROpenACCOpsInterfaces.cpp
  FIROpenACCTypeInterfaces.cpp
  FIROpenACCUtils.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  CUFDialect
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect

  LINK_LIBS
  CUFDialect
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
