add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp

  DEPENDS
  FIRSupport
  FIROpsIncGen

  LINK_LIBS
  FIRSupport
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
