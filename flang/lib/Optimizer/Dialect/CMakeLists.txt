add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp

  DEPENDS
  FIROpsIncGen

  LINK_LIBS
  FIRSupport
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIREDSC
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRStandardToLLVM
  MLIRTransforms

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
