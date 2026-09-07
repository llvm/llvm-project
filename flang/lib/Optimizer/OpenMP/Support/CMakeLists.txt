add_flang_library(FIROpenMPSupport
  FIROpenMPAttributes.cpp
  FIROpenMPOpsInterfaces.cpp
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
