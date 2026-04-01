add_aiir_upstream_c_api_library(AIIRCAPIAffine
  Affine.cpp
  AffinePasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRAffinePassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRAffineDialect
  AIIRAffineTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIAMDGPU
  AMDGPU.cpp
  AMDGPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRAMDGPUDialect
  AIIRAMDGPUTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIArith
  Arith.cpp
  ArithPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRArithDialect
  AIIRArithTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIArmNeon
  ArmNeon.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRArmNeonDialect
  AIIRArmNeonTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIArmSME
  ArmSME.cpp
  ArmSMEPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRArmSMEDialect
  AIIRArmSMETransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIArmSVE
  ArmSVE.cpp
  ArmSVEPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRArmSVEDialect
  AIIRArmSVETransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIAsync
  Async.cpp
  AsyncPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRAsyncPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRAsyncDialect
  AIIRAsyncTransforms
  AIIRPass
)

add_aiir_upstream_c_api_library(AIIRCAPIBufferization
  Bufferization.cpp
  BufferizationPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIComplex
  Complex.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR

  AIIRComplexDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIControlFlow
  ControlFlow.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRControlFlowDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIDLTI
  DLTI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRDLTIDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIEmitC
  EmitC.cpp
  EmitCPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIREmitCDialect
  AIIREmitCTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIFunc
  Func.cpp
  FuncPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRFuncDialect
  AIIRFuncTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIGPU
  GPU.cpp
  GPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRGPUPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRGPUTransforms
  AIIRPass
)

add_aiir_upstream_c_api_library(AIIRCAPIIndex
  Index.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRIndexDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIIRDL
  IRDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRIRDL
)

add_aiir_upstream_c_api_library(AIIRCAPILinalg
  Linalg.cpp
  LinalgPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRLinalgDialect
  AIIRPass
  AIIRLinalgTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPILLVM
  LLVM.cpp
  LLVMPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRLLVMDialect
  AIIRLLVMIRTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIMath
  Math.cpp
  MathPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRMathDialect
  AIIRMathTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIMemRef
  MemRef.cpp
  MemRefPasses.cpp

  PARTIAL_SOURCES_INTENDED
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRMemRefDialect
  AIIRMemRefTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIMLProgram
  MLProgram.cpp
  MLProgramPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRMLProgramPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRMLProgramDialect
  AIIRMLProgramTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIMPI
  MPI.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRMPIDialect
)

add_aiir_upstream_c_api_library(AIIRCAPINVGPU
  NVGPU.cpp
  NVGPUPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRNVGPUDialect
  AIIRNVGPUTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPINVVM
  NVVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRNVVMDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIOpenACC
  OpenACC.cpp
  OpenACCPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIROpenACCPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIROpenACCDialect
  AIIROpenACCTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIOpenMP
  OpenMP.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIROpenMPDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIPDL
  PDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRPDLDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIPDLInterp
  PDLInterp.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRPDLInterpDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIPtr
  Ptr.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRPtrDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIQuant
  Quant.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRQuantDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIROCDL
  ROCDL.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRROCDLDialect
)

add_aiir_upstream_c_api_library(AIIRCAPISCF
  SCF.cpp
  SCFPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRSCFPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRSCFDialect
  AIIRSCFTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIShape
  Shape.cpp
  ShapePasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRShapeDialect
  AIIRShapeOpsTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIShard
  Shard.cpp
  ShardPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRShardPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRShardDialect
  AIIRShardTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPISMT
  SMT.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRSMT
)

add_aiir_upstream_c_api_library(AIIRCAPISparseTensor
  SparseTensor.cpp
  SparseTensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRSparseTensorDialect
  AIIRSparseTensorTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPISPIRV
  SPIRV.cpp
  SPIRVPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRSPIRVPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRSPIRVDialect
  AIIRSPIRVTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPITensor
  Tensor.cpp
  TensorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRTensorDialect
  AIIRTensorTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPITosa
  Tosa.cpp
  TosaPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  AIIRTosaPassIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRTosaDialect
  AIIRTosaTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPITransformDialect
  Transform.cpp
  TransformPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRTransformDialect
  AIIRTransformDialectTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPITransformDialectTransforms
  TransformInterpreter.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRTransformDialectTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIUB
  UB.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRUBDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIVCIX
  VCIX.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRVCIXDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIVector
  Vector.cpp
  VectorPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRVectorDialect
  AIIRVectorTransforms
)

add_aiir_upstream_c_api_library(AIIRCAPIWasmSSA
  WasmSSA.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRWasmSSADialect
)

add_aiir_upstream_c_api_library(AIIRCAPIX86
  X86.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRX86Dialect
)

add_aiir_upstream_c_api_library(AIIRCAPIXeGPU
  XeGPU.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRXeGPUDialect
)

add_aiir_upstream_c_api_library(AIIRCAPIXeVM
  XeVM.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  AIIRCAPIIR
  AIIRXeVMDialect
)
