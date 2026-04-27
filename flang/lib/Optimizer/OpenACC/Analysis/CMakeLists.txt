add_flang_library(FIROpenACCAnalysis
  FIROpenACCSupportAnalysis.cpp

  DEPENDS
  CUFAttrs
  CUFDialect
  FIRAnalysis
  FIRDialect
  FIROpenACCSupport
  HLFIRDialect

  LINK_LIBS
  CUFAttrs
  CUFDialect
  FIRAnalysis
  FIRDialect
  FIROpenACCSupport
  HLFIRDialect

  MLIR_DEPS
  MLIROpenACCDialect
  MLIROpenACCUtils

  MLIR_LIBS
  MLIROpenACCDialect
  MLIROpenACCUtils
)

