add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp

  DEPENDS
  HLFIRDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  LINK_LIBS
  HLFIRDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  MLIR_DEPS
  MLIRDLTIDialect
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport

  MLIR_LIBS
  MLIRDLTIDialect
  MLIRIR
  MLIROpenACCDialect
  MLIRPass
  MLIRSupport
)
