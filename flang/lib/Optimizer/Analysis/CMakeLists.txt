add_flang_library(FIRAnalysis
  AliasAnalysis.cpp
  ArraySectionAnalyzer.cpp
  TBAAForest.cpp

  DEPENDS
  CUFDialect
  FIRDialect
  FIRSupport
  HLFIRDialect

  LINK_LIBS
  CUFDialect
  FIRDialect
  FIRSupport
  HLFIRDialect

  MLIR_DEPS
  MLIRIR
  MLIROpenMPDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathTransforms
  MLIROpenMPDialect
)
