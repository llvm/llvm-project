get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_flang_library(FIRSupport
  DataLayout.cpp
  InitFIR.cpp
  InternalNames.cpp

  DEPENDS
  FIROpsIncGen
  HLFIROpsIncGen
  MLIRIR
  ${dialect_libs}
  ${extension_libs}

  LINK_LIBS
  ${dialect_libs}
  ${extension_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport

  LINK_COMPONENTS
  TargetParser
)
