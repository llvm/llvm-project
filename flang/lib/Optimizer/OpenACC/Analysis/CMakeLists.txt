add_flang_library(FIROpenACCAnalysis
  FIROpenACCSupportAnalysis.cpp

  DEPENDS
  FIRAnalysis
  FIRDialect
  FIROpenACCSupport
  HLFIRDialect

  LINK_LIBS
  FIRAnalysis
  FIRDialect
  FIROpenACCSupport
  HLFIRDialect

  MLIR_DEPS
  MLIROpenACCDialect

  MLIR_LIBS
  MLIROpenACCDialect
)

