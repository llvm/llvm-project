add_clang_library(MLIRCIRTargetLowering
  CIRCXXABI.cpp
  LowerModule.cpp
  LowerItaniumCXXABI.cpp
  TargetLoweringInfo.cpp

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
