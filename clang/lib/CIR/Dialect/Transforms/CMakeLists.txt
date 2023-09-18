add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  MergeCleanups.cpp
  DropAST.cpp

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
  MLIRCIRASTAttrInterfaces
)
