add_flang_library(FIRCodeGen
  BoxedProcedure.cpp
  CGOps.cpp
  CodeGen.cpp
  CodeGenOpenMP.cpp
  FIROpPatterns.cpp
  PreCGRewrite.cpp
  TBAABuilder.cpp
  Target.cpp
  TargetRewrite.cpp
  TypeConverter.cpp

  DEPENDS
  CUFAttrs
  FIRDialect
  FIROptCodeGenPassIncGen
  CGOpsIncGen

  LINK_LIBS
  CUFAttrs
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIRSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
  TargetParser

  MLIR_LIBS
  MLIRComplexToLLVM
  MLIRComplexToStandard
  MLIRGPUDialect
  MLIRMathToFuncs
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMathToROCDL
  MLIROpenMPToLLVM
  MLIROpenACCDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRVectorToLLVM
)
