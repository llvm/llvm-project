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
  MLIROpenACCAnalysis
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRPass
  MLIRSupport

  MLIR_LIBS
  MLIRDLTIDialect
  MLIRIR
  MLIROpenACCAnalysis
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRPass
  MLIRSupport
)
