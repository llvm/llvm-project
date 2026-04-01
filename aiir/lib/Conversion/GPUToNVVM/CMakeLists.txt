set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
aiir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(AIIRGPUToNVVMIncGen)

add_aiir_conversion_library(AIIRGPUToNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp
  WmmaOpsToNvvm.cpp

  DEPENDS
  AIIRConversionPassIncGen
  AIIRGPUToNVVMIncGen

  LINK_LIBS PUBLIC
  AIIRArithToLLVM
  AIIRFuncToLLVM
  AIIRGPUDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMathToNVVM
  AIIRMemRefToLLVM
  AIIRNVGPUDialect
  AIIRNVGPUToNVVM
  AIIRNVVMDialect
  AIIRPass
  AIIRTransformUtils
  )
