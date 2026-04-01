# Exclude tests from libAIIR.so
add_aiir_library(AIIRLinalgTestPasses
  TestLinalgDecomposeOps.cpp
  TestLinalgDropUnitDims.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgRankReduceContractionOps.cpp
  TestLinalgTransforms.cpp
  TestPadFusion.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRLinalgTestPasses PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRArithTransforms
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRGPUTransforms
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRLLVMToLLVMIRTranslation
  AIIRMemRefDialect
  AIIRParser
  AIIRPass
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRTensorDialect
  AIIRTensorTransforms
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorToSCF
  AIIRVectorTransforms
  )
