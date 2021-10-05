add_flang_library(FIRSupport
  FIRContext.cpp
  InternalNames.cpp
  KindMapping.cpp

  DEPENDS
  ${dialect_libs}

  LINK_LIBS
  ${dialect_libs}
  MLIROpenMPToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
)
