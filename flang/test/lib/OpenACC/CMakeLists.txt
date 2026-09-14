add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp
  TestOpenACCSupport.cpp

  DEPENDS
  HLFIRDialect
  FIRDialect
  FIROpenACCAnalysis
  FIROpenACCSupport
  FIRSupport

  LINK_LIBS
  HLFIRDialect
  FIRDialect
  FIROpenACCAnalysis
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
