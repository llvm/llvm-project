add_flang_library(FIRAnalysis
  AliasAnalysis.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport

  LINK_LIBS
  FIRBuilder
  FIRDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  FIRSupport
)
