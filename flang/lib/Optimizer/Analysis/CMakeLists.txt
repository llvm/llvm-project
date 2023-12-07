add_flang_library(FIRAnalysis
  AliasAnalysis.cpp

  DEPENDS
  FIRDialect
  HLFIRDialect
  MLIRIR

  LINK_LIBS
  FIRBuilder
  FIRDialect
  HLFIRDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  FIRSupport
)
