add_flang_library(FIRCodeGen
  BoxedProcedure.cpp
  CGOps.cpp
  CodeGen.cpp
  PreCGRewrite.cpp
  Target.cpp
  TargetRewrite.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptCodeGenPassIncGen
  CGOpsIncGen

  LINK_LIBS
  FIRBuilder
  FIRDialect
  FIRSupport
  MLIRMathToFuncs
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
