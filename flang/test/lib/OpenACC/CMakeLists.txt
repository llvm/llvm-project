add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp

  DEPENDS
  FIRDialect
  FIROpenACCSupport
  FIRSupport
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport

  LINK_LIBS
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  MLIR_LIBS
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport
)
