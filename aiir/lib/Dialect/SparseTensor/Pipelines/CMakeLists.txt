add_aiir_dialect_library(AIIRSparseTensorPipelines
  SparseTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  AIIRArithTransforms
  AIIRAffineToStandard
  AIIRBufferizationTransforms
  AIIRComplexToLLVM
  AIIRComplexToLibm
  AIIRComplexToStandard
  AIIRFuncTransforms
  AIIRGPUToNVVMTransforms
  AIIRGPUTransforms
  AIIRLinalgTransforms
  AIIRMathToLibm
  AIIRMathToLLVM
  AIIRMemRefToLLVM
  AIIRPass
  AIIRReconcileUnrealizedCasts
  AIIRSCFToControlFlow
  AIIRSparseTensorDialect
  AIIRSparseTensorTransforms
  AIIRTensorTransforms
  AIIRUBToLLVM
  AIIRVectorToLLVMPass
  AIIRVectorTransforms
)
