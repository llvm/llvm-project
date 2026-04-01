add_aiir_dialect_library(AIIRTargetLLVMIRTransforms
  TargetToDataLayout.cpp
  TargetToTargetFeatures.cpp
  TargetUtils.cpp

  DEPENDS
  AIIRTargetLLVMIRTransformsIncGen

  LINK_COMPONENTS
  MC
  Target
  TargetParser
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PUBLIC
  AIIRDLTIDialect
  AIIRLLVMDialect
  AIIRPass
  AIIRTargetLLVMIRImport
  )
