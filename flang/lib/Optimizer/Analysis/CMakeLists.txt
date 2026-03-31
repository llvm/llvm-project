add_flang_library(FIRAnalysis
  AliasAnalysis.cpp
  ArraySectionAnalyzer.cpp
  TBAAForest.cpp

  DEPENDS
  CUFDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport
  HLFIRDialect

  LINK_LIBS
  CUFDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport
  HLFIRDialect

  MLIR_DEPS
  MLIRIR
  MLIROpenACCDialect
  MLIROpenMPDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  MLIROpenACCDialect
  MLIROpenMPDialect
)
