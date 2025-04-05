add_flang_library(FIRCodeGenDialect
  CGOps.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  CGOpsIncGen

  LINK_LIBS
  FIRDialect
  MLIRIR

  LINK_COMPONENTS
  AsmParser
  AsmPrinter

  )

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

  PARTIAL_SOURCES_INTENDED

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
