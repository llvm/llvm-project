add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysis
  MLIRSideEffectInterfaces
)
