add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysisBase
  MLIRSideEffectInterfaces
)
