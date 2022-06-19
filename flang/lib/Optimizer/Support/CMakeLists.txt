get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRSupport
  FIRContext.cpp
  InitFIR.cpp
  InternalNames.cpp
  KindMapping.cpp

  DEPENDS
  FIROpsIncGen
  MLIRIR
  ${dialect_libs}

  LINK_LIBS
  ${dialect_libs}
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
)
