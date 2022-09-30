add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  Inliner.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  FIRSupport
  FIROpsIncGen

  LINK_LIBS
  FIRSupport
  MLIRArithDialect
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
