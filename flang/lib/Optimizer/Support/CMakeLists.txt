get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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
  ${dialect_libs}
  ${extension_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)
