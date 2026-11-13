add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp
  CIRConstantFolding.cpp
  CIRFunctionExitAnalysis.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysisBase
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
)
