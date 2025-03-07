add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp

  DEPENDS
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  LINK_LIBS
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  MLIR_DEPS
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport

  MLIR_LIBS
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport
)
