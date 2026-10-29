add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp

  DEPENDS
  MLIRPtrMemorySpaceInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysisBase
  MLIRSideEffectInterfaces
)
