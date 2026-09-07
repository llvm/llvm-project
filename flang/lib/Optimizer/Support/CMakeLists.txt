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

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)
