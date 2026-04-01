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

  AIIR_DEPS
  AIIRIR
  AIIROpenMPDialect

  AIIR_LIBS
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRMathTransforms
  AIIROpenMPDialect
)
