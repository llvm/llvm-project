add_clang_library(MLIRCIRTargetLowering
  ABIInfo.cpp
  CIRCXXABI.cpp
  CIRLowerContext.cpp
  LowerModule.cpp
  LowerTypes.cpp
  LowerItaniumCXXABI.cpp
  TargetLoweringInfo.cpp
  Targets/X86.cpp
  
  DEPENDS
  clangBasic

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC

  clangBasic
  MLIRIR
  MLIRPass
  MLIRDLTIDialect
  MLIRCIR
  MLIRCIRInterfaces
  MLIRLLVMDialect
)
