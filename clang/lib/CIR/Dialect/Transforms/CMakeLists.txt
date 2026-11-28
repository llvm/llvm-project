add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  CallConvLoweringPass.cpp
  CIRCanonicalize.cpp
  CIRFoldEmptyBranches.cpp
  CIRSimplify.cpp
  CIRTransformUtils.cpp
  CXXABILowering.cpp
  EHABILowering.cpp
  TargetLowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  GotoElimination.cpp
  GotoSolver.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  RecordTypeConverter.cpp
  CIRABIClassifier.cpp
  CIRABIClassifierAArch64.cpp
  CIRABIClassifierAMDGPU.cpp
  CIRABIClassifierSPIRV.cpp
  CIRABIClassifierX86_64.cpp

  DEPENDS
  MLIRCIRPassIncGen
  mlir-headers

  LINK_COMPONENTS
  ABI
  TargetParser

  LINK_LIBS PUBLIC
  clangBasic

  MLIRABI
  MLIRAnalysisBase
  MLIRDLTIDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
  MLIRCIRTargetLowering
  MLIROpenACCDialect
  MLIROpenMPDialect
)
