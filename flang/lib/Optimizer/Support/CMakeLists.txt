add_flang_library(FIRSupport
  DataLayout.cpp
  InitFIR.cpp
  InternalNames.cpp
  Utils.cpp

  DEPENDS
  FIROpsIncGen
  HLFIROpsIncGen

  LINK_LIBS
  FIRDialect

  LINK_COMPONENTS
  TargetParser

  MLIR_DEPS
  MLIRIR
  ${dialect_libs}
  ${extension_libs}

  MLIR_LIBS
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)
