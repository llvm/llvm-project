add_flang_library(FIRSupport
  FIRContext.cpp
  InternalNames.cpp
  KindMapping.cpp

  DEPENDS
  ${dialect_libs}
  MLIRIR

  LINK_LIBS
  ${dialect_libs}
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
)
