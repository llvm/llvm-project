add_flang_library(FIRSupport
  DataLayout.cpp
  InitFIR.cpp
  InternalNames.cpp
  Utils.cpp

  DEPENDS
  FIROpsIncGen
  HLFIROpsIncGen
  MIFOpsIncGen

  LINK_LIBS
  FIRDialect
  MIFDialect

  LINK_COMPONENTS
  TargetParser

  AIIR_DEPS
  AIIRIR
  ${dialect_libs}
  ${extension_libs}

  AIIR_LIBS
  AIIRBuiltinToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
  AIIROpenACCToLLVMIRTranslation
  AIIROpenMPToLLVMIRTranslation
  AIIRRegisterAllDialects
  AIIRRegisterAllExtensions
  AIIRTargetLLVMIRExport
  AIIRTargetLLVMIRImport
)
