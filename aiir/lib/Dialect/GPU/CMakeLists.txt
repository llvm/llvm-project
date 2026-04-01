add_aiir_dialect_library(AIIRGPUDialect
  IR/GPUDialect.cpp
  IR/InferIntRangeInterfaceImpls.cpp
  IR/ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/GPU

  DEPENDS
  AIIRGPUOpsIncGen
  AIIRGPUOpsAttributesIncGen
  AIIRGPUOpsEnumsGen
  AIIRGPUOpInterfacesIncGen
  AIIRGPUCompilationAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDLTIDialect
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRInferIntRangeInterface
  AIIRIR
  AIIRMathDialect
  AIIRMemRefDialect
  AIIRSideEffectInterfaces
  AIIRSupport
  )

add_aiir_dialect_library(AIIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/BufferDeallocationOpInterfaceImpl.cpp
  Transforms/DecomposeMemRefs.cpp
  Transforms/EliminateBarriers.cpp
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ModuleToBinary.cpp
  Transforms/NVVMAttachTarget.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/PromoteShuffleToAMDGPU.cpp
  Transforms/ROCDLAttachTarget.cpp
  Transforms/SPIRVAttachTarget.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SubgroupIdRewriter.cpp
  Transforms/SubgroupReduceLowering.cpp
  Transforms/XeVMAttachTarget.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/GPU

  DEPENDS
  AIIRGPUPassIncGen
  AIIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  AIIRAMDGPUDialect
  AIIRAffineUtils
  AIIRArithDialect
  AIIRAsyncDialect
  AIIRBufferizationDialect
  AIIRDataLayoutInterfaces
  AIIRExecutionEngineUtils
  AIIRGPUDialect
  AIIRGPUUtils
  AIIRIR
  AIIRIndexDialect
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRNVVMTarget
  AIIRPass
  AIIRROCDLDialect
  AIIRROCDLTarget
  AIIRSCFDialect
  AIIRSPIRVTarget
  AIIRSideEffectInterfaces
  AIIRSupport
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRXeVMDialect
  )

add_subdirectory(TransformOps)
add_subdirectory(Pipelines)
add_subdirectory(Utils)
