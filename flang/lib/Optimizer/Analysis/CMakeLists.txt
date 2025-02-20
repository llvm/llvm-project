add_flang_library(FIRAnalysis
  AliasAnalysis.cpp
  TBAAForest.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  HLFIRDialect
  FIRCodeGen

  LINK_LIBS
  FIRBuilder
  FIRDialect
  FIRSupport
  HLFIRDialect
  FIRCodeGen

  MLIR_DEPS
  MLIRIR
  MLIROpenMPDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  MLIROpenMPDialect
)
