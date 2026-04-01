if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(AMDGPU_LIBS
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_aiir_conversion_library(AIIRGPUToGPURuntimeTransforms
  AttrToSPIRVConverter.cpp
  GPUToLLVMConversion.cpp
  GPUOpsLowering.cpp
  IndexIntrinsicsOpLowering.cpp

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core
  MC
  ${AMDGPU_LIBS}
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  AIIRArithToLLVM
  AIIRAsyncToLLVM
  AIIRControlFlowToLLVM
  AIIRConvertToLLVMPass
  AIIRFuncToLLVM
  AIIRGPUTransforms
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefToLLVM
  AIIRPass
  AIIRSupport
  AIIRVectorToLLVM
)
