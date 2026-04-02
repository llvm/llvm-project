add_flang_library(FIRCodeGen
  BoxedProcedure.cpp
  CodeGen.cpp
  CodeGenOpenMP.cpp
  FIROpPatterns.cpp
  LLVMInsertChainFolder.cpp
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
  FIROpenACCSupport
  FIROpenMPSupport
  FIRSupport

  LINK_COMPONENTS
  Remarks
  TargetParser

  MLIR_LIBS
  MLIRComplexToLLVM
  MLIRComplexToROCDLLibraryCalls
  MLIRComplexToStandard
  MLIRGPUDialect
  MLIRIndexToLLVM
  MLIRMathToFuncs
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMathToNVVM
  MLIRMathToROCDL
  MLIROpenMPToLLVM
  MLIROpenACCDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRVectorToLLVM
)

# FIROpPatterns.cpp and PreCGRewrite.cpp each redefine local helpers
# (getTypeDescFieldId, DeclareOpConversion) that clash in unity builds.
set_source_files_properties(
  FIROpPatterns.cpp
  PreCGRewrite.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
