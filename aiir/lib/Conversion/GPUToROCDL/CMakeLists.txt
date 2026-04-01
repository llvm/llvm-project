set(LLVM_TARGET_DEFINITIONS GPUToROCDL.td)
aiir_tablegen(GPUToROCDL.cpp.inc -gen-rewriters)
add_public_tablegen_target(AIIRGPUToROCDLIncGen)

add_aiir_conversion_library(AIIRGPUToROCDLTransforms
  LowerGpuOpsToROCDLOps.cpp

  DEPENDS
  AIIRConversionPassIncGen
  AIIRGPUToROCDLIncGen

  LINK_LIBS PUBLIC
  AIIRArithToLLVM
  AIIRArithTransforms
  AIIRMathToLLVM
  AIIRMathToROCDL
  AIIRAMDGPUToROCDL
  AIIRFuncToLLVM
  AIIRGPUDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefToLLVM
  AIIRROCDLDialect
  AIIRPass
  )
