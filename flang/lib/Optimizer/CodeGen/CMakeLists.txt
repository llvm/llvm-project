add_flang_library(FIRCodeGen
  BoxedProcedure.cpp
  CodeGen.cpp
  CodeGenOpenMP.cpp
  FIROpPatterns.cpp
  LowerRepackArrays.cpp
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
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport

  LINK_COMPONENTS
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
