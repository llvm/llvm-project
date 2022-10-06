add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  MergeCleanups.cpp
  DropAST.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC

  MLIRAnalysis
  MLIRIR
  MLIRCIR
  MLIRPass
  MLIRTransformUtils
)
