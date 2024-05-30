add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  LoweringPrepareItaniumCXXABI.cpp
  LoweringPrepareAArch64CXXABI.cpp
  MergeCleanups.cpp
  DropAST.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  StdHelpers.cpp
  FlattenCFG.cpp
  GotoSolver.cpp
  SCFPrepare.cpp
  CallConvLowering.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic

  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
)
