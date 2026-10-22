add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  CallConvLoweringPass.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  CIRTransformUtils.cpp
  CXXABILowering.cpp
  EHABILowering.cpp
  TargetLowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  GotoSolver.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  RecordTypeConverter.cpp
  CIRABIClassifier.cpp
  CIRABIClassifierAMDGPU.cpp
  CIRABIClassifierX86_64.cpp

  DEPENDS
  MLIRCIRPassIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBytecodeOpInterfaceIncGen

  LINK_COMPONENTS
  ABI

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
