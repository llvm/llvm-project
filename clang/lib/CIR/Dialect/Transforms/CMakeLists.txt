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

  DEPENDS
  MLIRCIRPassIncGen

  LINK_COMPONENTS
  ABI

  LINK_LIBS PUBLIC
  clangBasic

  MLIRABI
  MLIRAnalysisUtilities
  MLIRDLTIDialect
  MLIRIRCore
  MLIRPass
  MLIRRewriteUtils

  MLIRCIR
  MLIRCIRInterfacesCore
  MLIRCIRTargetLowering
  MLIROpenACCDialect
  MLIROpenMPDialect
)
