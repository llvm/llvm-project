add_subdirectory(TargetLowering)

add_clang_library(AIIRCIRTransforms
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  CXXABILowering.cpp
  EHABILowering.cpp
  TargetLowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  GotoSolver.cpp
  IdiomRecognizer.cpp

  DEPENDS
  AIIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic

  AIIRAnalysis
  AIIRIR
  AIIRPass
  AIIRTransformUtils

  AIIRCIR
  AIIRCIRInterfaces
  AIIRCIRTargetLowering
)
