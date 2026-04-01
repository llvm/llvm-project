add_aiir_dialect_library(AIIRGPUPipelines
  GPUToNVVMPipeline.cpp
  GPUToXeVMPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/GPU

  LINK_LIBS PUBLIC
  AIIRMemRefTransforms
  AIIRFuncDialect
  AIIRPass
  AIIRTransforms
  AIIRLinalgTransforms
  AIIRAffineToStandard
  AIIRGPUToLLVMSPV
  AIIRGPUToNVVMTransforms
  AIIRIndexToLLVM
  AIIRMathToLLVM
  AIIRMathToXeVM
  AIIRNVGPUToNVVM
  AIIRNVVMToLLVM
  AIIRReconcileUnrealizedCasts
  AIIRSCFToControlFlow
  AIIRVectorToLLVMPass
  AIIRVectorToSCF
  AIIRXeGPUTransforms
  AIIRXeGPUToXeVM
  AIIRXeVMToLLVM
)
