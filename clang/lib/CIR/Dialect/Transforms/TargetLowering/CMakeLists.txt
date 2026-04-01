add_clang_library(AIIRCIRTargetLowering
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
  AIIRIR
  AIIRPass
  AIIRDLTIDialect
  AIIRCIR
  AIIRCIRInterfaces
  AIIRLLVMDialect
)
