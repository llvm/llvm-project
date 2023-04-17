get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRSupport
  InitFIR.cpp
  InternalNames.cpp
  TypeCode.cpp

  DEPENDS
  FIROpsIncGen
  HLFIROpsIncGen
  MLIRIR
  ${dialect_libs}

  LINK_LIBS
  ${dialect_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  LINK_COMPONENTS
  TargetParser
)
