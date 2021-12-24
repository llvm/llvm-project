add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  FIRSupport
  FIROpsIncGen

  LINK_LIBS
  FIRSupport
  MLIRArithmetic
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
