add_clang_library(MLIRCIRTransforms
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp

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
