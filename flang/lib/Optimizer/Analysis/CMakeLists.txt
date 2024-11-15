add_flang_library(FIRAnalysis
  AliasAnalysis.cpp
  TBAAForest.cpp

  DEPENDS
  FIRDialect
  HLFIRDialect
  MLIRIR
  MLIROpenMPDialect

  LINK_LIBS
  FIRBuilder
  FIRDialect
  HLFIRDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  MLIROpenMPDialect
  FIRSupport
)
