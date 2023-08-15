add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  MergeCleanups.cpp
  DropAST.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST

  MLIRAnalysis
  MLIRIR
  MLIRCIR
  MLIRPass
  MLIRTransformUtils
)
