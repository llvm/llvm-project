
add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  Inliner.cpp

  DEPENDS
  FIRSupport
  FIROpsIncGen

  LINK_LIBS
  FIRSupport
  MLIROpenMPToLLVM
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
