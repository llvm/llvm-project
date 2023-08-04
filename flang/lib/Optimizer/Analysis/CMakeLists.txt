add_flang_library(FIRAnalysis
  AliasAnalysis.cpp

  DEPENDS
  FIRDialect
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
