add_flang_library(FIRAnalysis
  AliasAnalysis.cpp
  TBAAForest.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  HLFIRDialect
  MLIRIR
  MLIROpenMPDialect

  LINK_LIBS
  FIRBuilder
  FIRDialect
  HLFIRDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  MLIROpenMPDialect
  FIRSupport
)
