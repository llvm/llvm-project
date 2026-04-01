include(AddAIIRPython)

# Specifies that all AIIR packages are co-located under the `AIIR_PYTHON_PACKAGE_PREFIX.`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("AIIR_PYTHON_PACKAGE_PREFIX=${AIIR_PYTHON_PACKAGE_PREFIX}.")
set(AIIRPythonModules_ROOT_PREFIX "${AIIR_BINARY_DIR}/${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}")
set(PYTHON_SOURCE_DIR "${AIIR_SOURCE_DIR}/lib/Bindings/Python")

################################################################################
# Structural groupings.
################################################################################

declare_aiir_python_sources(AIIRPythonSources)
declare_aiir_python_sources(AIIRPythonSources.Dialects
  ADD_TO_PARENT AIIRPythonSources)
declare_aiir_python_sources(AIIRPythonSources.Core
  ADD_TO_PARENT AIIRPythonSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_aiir_python_sources(AIIRPythonSources.Core.Python
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  ADD_TO_PARENT AIIRPythonSources.Core
  SOURCES
    _aiir_libs/__init__.py
    _aiir_libs/_aiir/py.typed
    ir.py
    passmanager.py
    rewrite.py
    dialects/_ods_common.py
    dialects/ext.py
)

declare_aiir_python_sources(AIIRPythonSources.Core.Python.Extras
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  ADD_TO_PARENT AIIRPythonSources.Core.Python
  SOURCES
    extras/types.py
    extras/meta.py
)

declare_aiir_python_sources(AIIRPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  ADD_TO_PARENT AIIRPythonSources
  SOURCES
    execution_engine.py
    _aiir_libs/_aiirExecutionEngine.pyi
  SOURCES_GLOB
    runtime/*.py
)

declare_aiir_python_sources(AIIRPythonCAPI.HeaderSources
  ROOT_DIR "${AIIR_SOURCE_DIR}/include"
  SOURCES_GLOB "aiir-c/*.h"
)

################################################################################
# Dialect bindings
################################################################################

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/AffineOps.td
  SOURCES
    dialects/affine.py
  DIALECT_NAME affine
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/AMDGPUOps.td
  SOURCES
    dialects/amdgpu.py
  DIALECT_NAME amdgpu
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/AsyncOps.td
  SOURCES_GLOB dialects/async_dialect/*.py
  DIALECT_NAME async)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/BufferizationOps.td
  SOURCES
    dialects/bufferization.py
  DIALECT_NAME bufferization
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/aiir/Dialect/Bufferization/IR/BufferizationEnums.td"
)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/BuiltinOps.td
  SOURCES
    dialects/builtin.py
  DIALECT_NAME builtin)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ComplexOps.td
  SOURCES
    dialects/complex.py
  DIALECT_NAME complex)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/IndexOps.td
  SOURCES
    dialects/index.py
  DIALECT_NAME index
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ControlFlowOps.td
  SOURCES
    dialects/cf.py
  DIALECT_NAME cf)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/FuncOps.td
  SOURCES
    dialects/func.py
  DIALECT_NAME func)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/OpenACCOps.td
  SOURCES
    dialects/openacc.py
  DIALECT_NAME acc
  DEPENDS acc_common_td
  )

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/GPUOps.td
  SOURCES_GLOB dialects/gpu/*.py
  DIALECT_NAME gpu
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/LinalgOps.td
  SOURCES
  SOURCES_GLOB
    dialects/linalg/*.py
  DIALECT_NAME linalg
  DEPENDS LinalgOdsGen
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/LLVMOps.td
  SOURCES
    dialects/llvm.py
  DIALECT_NAME llvm
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_extension_python_bindings(
ADD_TO_PARENT AIIRPythonSources.Dialects
ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TransformPDLExtensionOps.td
  SOURCES
    dialects/transform/pdl.py
  DIALECT_NAME transform
  EXTENSION_NAME transform_pdl_extension)

declare_aiir_dialect_extension_python_bindings(
ADD_TO_PARENT AIIRPythonSources.Dialects
ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TransformSMTExtensionOps.td
  SOURCES
    dialects/transform/smt.py
  DIALECT_NAME transform
  EXTENSION_NAME transform_smt_extension)

declare_aiir_dialect_extension_python_bindings(
ADD_TO_PARENT AIIRPythonSources.Dialects
ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TransformDebugExtensionOps.td
  SOURCES
    dialects/transform/debug.py
  DIALECT_NAME transform
  EXTENSION_NAME transform_debug_extension)

declare_aiir_dialect_extension_python_bindings(
ADD_TO_PARENT AIIRPythonSources.Dialects
ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TransformTuneExtensionOps.td
  SOURCES
    dialects/transform/tune.py
  DIALECT_NAME transform
  EXTENSION_NAME transform_tune_extension)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TransformOps.td
  SOURCES
    dialects/transform/__init__.py
    _aiir_libs/_aiir/dialects/transform/__init__.pyi
  DIALECT_NAME transform
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/aiir/Dialect/Transform/IR/TransformAttrs.td"
)

declare_aiir_python_sources(
  AIIRPythonSources.Dialects.transform.extras
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  GEN_ENUM_BINDINGS
  SOURCES
    dialects/transform/extras/__init__.py)

declare_aiir_python_sources(
  AIIRPythonSources.Dialects.transform.interpreter
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  SOURCES
    dialects/transform/interpreter/__init__.py)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/BufferizationTransformOps.td
  SOURCES
    dialects/transform/bufferization.py
  DIALECT_NAME transform
  EXTENSION_NAME bufferization_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/GPUTransformOps.td
  SOURCES
    dialects/transform/gpu.py
  DIALECT_NAME transform
  EXTENSION_NAME gpu_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SCFLoopTransformOps.td
  SOURCES
    dialects/transform/loop.py
  DIALECT_NAME transform
  EXTENSION_NAME loop_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/MemRefTransformOps.td
  SOURCES
    dialects/transform/memref.py
  DIALECT_NAME transform
  EXTENSION_NAME memref_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/NVGPUTransformOps.td
  SOURCES
    dialects/transform/nvgpu.py
  DIALECT_NAME transform
  EXTENSION_NAME nvgpu_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/LinalgStructuredTransformOps.td
  SOURCES
    dialects/transform/structured.py
  DIALECT_NAME transform
  EXTENSION_NAME structured_transform
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/aiir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SparseTensorTransformOps.td
  SOURCES
    dialects/transform/sparse_tensor.py
  DIALECT_NAME transform
  EXTENSION_NAME sparse_tensor_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TensorTransformOps.td
  SOURCES
    dialects/transform/tensor.py
  DIALECT_NAME transform
  EXTENSION_NAME tensor_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/VectorTransformOps.td
  SOURCES
    dialects/transform/vector.py
  DIALECT_NAME transform
  EXTENSION_NAME vector_transform
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/aiir/Dialect/Vector/Transforms/VectorTransformsBase.td"
)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/X86TransformOps.td
  SOURCES
    dialects/transform/x86.py
  DIALECT_NAME transform
  EXTENSION_NAME x86_transform)

declare_aiir_dialect_extension_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/XeGPUTransformOps.td
  SOURCES
    dialects/transform/xegpu.py
  DIALECT_NAME transform
  EXTENSION_NAME xegpu_transform)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/MathOps.td
  SOURCES dialects/math.py
  DIALECT_NAME math)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ArithOps.td
  SOURCES
    dialects/arith.py
  DIALECT_NAME arith
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/MemRefOps.td
  SOURCES
    dialects/memref.py
  DIALECT_NAME memref)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ShardOps.td
  SOURCES
    dialects/shard.py
  DIALECT_NAME shard
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/MLProgramOps.td
  SOURCES
    dialects/ml_program.py
  DIALECT_NAME ml_program)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/NVGPUOps.td
  SOURCES
    dialects/nvgpu.py
  DIALECT_NAME nvgpu
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/NVVMOps.td
  SOURCES
    dialects/nvvm.py
  DIALECT_NAME nvvm
  GEN_ENUM_BINDINGS)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ROCDLOps.td
  SOURCES
    dialects/rocdl.py
  DIALECT_NAME rocdl)

declare_aiir_python_sources(
  AIIRPythonSources.Dialects.quant
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  GEN_ENUM_BINDINGS
  SOURCES
    dialects/quant.py
    _aiir_libs/_aiir/dialects/quant.pyi)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/EmitC.td
  SOURCES
    dialects/emitc.py
  DIALECT_NAME emitc)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/PDLOps.td
  SOURCES
    dialects/pdl.py
    _aiir_libs/_aiir/dialects/pdl.pyi
  DIALECT_NAME pdl)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/OpenMPOps.td
  SOURCES
    dialects/openmp.py
  DIALECT_NAME omp
  DEPENDS omp_common_td)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
  DIALECT_NAME scf)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/ShapeOps.td
  SOURCES dialects/shape.py
  DIALECT_NAME shape)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SparseTensorOps.td
  SOURCES dialects/sparse_tensor.py
  DIALECT_NAME sparse_tensor
  GEN_ENUM_BINDINGS_TD_FILE
    "../../include/aiir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td"
)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SMTOps.td
  GEN_ENUM_BINDINGS
  SOURCES
    dialects/smt.py
  DIALECT_NAME smt)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/SPIRVOps.td
  SOURCES dialects/spirv.py
  DIALECT_NAME spirv)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TensorOps.td
  SOURCES
    dialects/tensor.py
  DIALECT_NAME tensor)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/TosaOps.td
  SOURCES dialects/tosa.py
  DIALECT_NAME tosa
)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/UBOps.td
  SOURCES dialects/ub.py
  DIALECT_NAME ub
)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/VectorOps.td
  SOURCES dialects/vector.py
  DIALECT_NAME vector
  GEN_ENUM_BINDINGS_TD_FILE
    "dialects/VectorAttributes.td")

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/X86.td
  SOURCES dialects/x86.py
  DIALECT_NAME x86)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT AIIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
  TD_FILE dialects/IRDLOps.td
  SOURCES dialects/irdl.py
  DIALECT_NAME irdl
  GEN_ENUM_BINDINGS
)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

declare_aiir_python_extension(AIIRPythonExtension.Core
  MODULE_NAME _aiir
  ADD_TO_PARENT AIIRPythonSources.Core
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRInterfaces.cpp
    Pass.cpp
    Rewrite.cpp

    # Headers must be included explicitly so they are installed.
    Pass.h
    Rewrite.h
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIDebug
    AIIRCAPIIR
    AIIRCAPIInterfaces

    # Dialects
    AIIRCAPIFunc
)

# This extension exposes an API to register all dialects, extensions, and passes
# packaged in upstream AIIR and it is used for the upstream "aiir" Python
# package. Downstreams will likely want to provide their own and not depend
# on this one, since it links in the world.
# Note that this is not added to any top-level source target for transitive
# inclusion: It must be included explicitly by downstreams if desired. Note that
# this has a very large impact on what gets built/packaged.
declare_aiir_python_extension(AIIRPythonExtension.RegisterEverything
  MODULE_NAME _aiirRegisterEverything
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIConversion
    AIIRCAPITransforms
    AIIRCAPIRegisterEverything
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.Linalg.Nanobind
  MODULE_NAME _aiirDialectsLinalg
  ADD_TO_PARENT AIIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectLinalg.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPILinalg
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.GPU.Nanobind
  MODULE_NAME _aiirDialectsGPU
  ADD_TO_PARENT AIIRPythonSources.Dialects.gpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectGPU.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIGPU
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.LLVM.Nanobind
  MODULE_NAME _aiirDialectsLLVM
  ADD_TO_PARENT AIIRPythonSources.Dialects.llvm
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectLLVM.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPILLVM
    # Misnomer - this is only the LLVMIR translation target.
    AIIRCAPITarget
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.Quant.Nanobind
  MODULE_NAME _aiirDialectsQuant
  ADD_TO_PARENT AIIRPythonSources.Dialects.quant
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectQuant.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIQuant
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.NVGPU.Nanobind
  MODULE_NAME _aiirDialectsNVGPU
  ADD_TO_PARENT AIIRPythonSources.Dialects.nvgpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectNVGPU.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPINVGPU
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.PDL.Nanobind
  MODULE_NAME _aiirDialectsPDL
  ADD_TO_PARENT AIIRPythonSources.Dialects.pdl
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectPDL.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIPDL
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.SparseTensor.Nanobind
  MODULE_NAME _aiirDialectsSparseTensor
  ADD_TO_PARENT AIIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectSparseTensor.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPISparseTensor
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.Transform.Nanobind
  MODULE_NAME _aiirDialectsTransform
  ADD_TO_PARENT AIIRPythonSources.Dialects.transform
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectTransform.cpp
    Rewrite.h
  PRIVATE_LINK_LIBS
    LLVMSupport
    AIIRPythonExtension.Core
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPITransformDialect
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.IRDL.Nanobind
  MODULE_NAME _aiirDialectsIRDL
  ADD_TO_PARENT AIIRPythonSources.Dialects.irdl
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectIRDL.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIIRDL
)

declare_aiir_python_extension(AIIRPythonExtension.AsyncDialectPasses
  MODULE_NAME _aiirAsyncPasses
  ADD_TO_PARENT AIIRPythonSources.Dialects.async
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    AsyncPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIAsync
)

if(AIIR_ENABLE_EXECUTION_ENGINE)
  declare_aiir_python_extension(AIIRPythonExtension.ExecutionEngine
    MODULE_NAME _aiirExecutionEngine
    ADD_TO_PARENT AIIRPythonSources.ExecutionEngine
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    SOURCES
      ExecutionEngineModule.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      AIIRCAPIExecutionEngine
  )
endif()

declare_aiir_python_extension(AIIRPythonExtension.GPUDialectPasses
  MODULE_NAME _aiirGPUPasses
  ADD_TO_PARENT AIIRPythonSources.Dialects.gpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    GPUPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIGPU
)

declare_aiir_python_extension(AIIRPythonExtension.LinalgPasses
  MODULE_NAME _aiirLinalgPasses
  ADD_TO_PARENT AIIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    LinalgPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPILinalg
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.SMT.Nanobind
  MODULE_NAME _aiirDialectsSMT
  ADD_TO_PARENT AIIRPythonSources.Dialects.smt
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectSMT.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPISMT
    AIIRCAPIExportSMTLIB
)

declare_aiir_python_extension(AIIRPythonExtension.SparseTensorDialectPasses
  MODULE_NAME _aiirSparseTensorPasses
  ADD_TO_PARENT AIIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    SparseTensorPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPISparseTensor
)

declare_aiir_python_extension(AIIRPythonExtension.TransformInterpreter
  MODULE_NAME _aiirTransformInterpreter
  ADD_TO_PARENT AIIRPythonSources.Dialects.transform
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    TransformInterpreter.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPITransformDialectTransforms
)

declare_aiir_python_extension(AIIRPythonExtension.Dialects.AMDGPU.Nanobind
  MODULE_NAME _aiirDialectsAMDGPU
  ADD_TO_PARENT AIIRPythonSources.Dialects.amdgpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectAMDGPU.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIAMDGPU
)


# TODO: Figure out how to put this in the test tree.
# This should not be included in the main Python extension. However,
# putting it into AIIRPythonTestSources along with the dialect declaration
# above confuses Python module loader when running under lit.
set(_ADDL_TEST_SOURCES)
if(AIIR_INCLUDE_TESTS)
  set(_ADDL_TEST_SOURCES AIIRPythonTestSources)
  declare_aiir_python_sources(AIIRPythonTestSources)
  declare_aiir_python_sources(AIIRPythonTestSources.Dialects
    ADD_TO_PARENT AIIRPythonTestSources)

  # TODO: this uses a tablegen file from the test directory and should be
  # decoupled from here.
  declare_aiir_python_sources(
    AIIRPythonTestSources.Dialects.PythonTest
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir"
    ADD_TO_PARENT AIIRPythonTestSources.Dialects
    SOURCES
      dialects/python_test.py
  )
  set(LLVM_TARGET_DEFINITIONS
    "${AIIR_MAIN_SRC_DIR}/test/python/python_test_ops.td")
  aiir_tablegen(
    "dialects/_python_test_ops_gen.py"
    -gen-python-op-bindings
    -bind-dialect=python_test)
  add_public_tablegen_target(PythonTestDialectPyIncGen)
  declare_aiir_python_sources(
    AIIRPythonTestSources.Dialects.PythonTest.ops_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    ADD_TO_PARENT AIIRPythonTestSources.Dialects.PythonTest
    SOURCES "dialects/_python_test_ops_gen.py")

  declare_aiir_python_extension(AIIRPythonTestSources.PythonTestExtensionNanobind
    MODULE_NAME _aiirPythonTestNanobind
    ADD_TO_PARENT AIIRPythonTestSources.Dialects
    ROOT_DIR "${AIIR_SOURCE_DIR}/test/python/lib"
    SOURCES
      PythonTestModuleNanobind.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      AIIRCAPIPythonTestDialect
  )
endif()

declare_aiir_python_extension(AIIRPythonExtension.AIIRPythonSupport
  _PRIVATE_SUPPORT_LIB
  MODULE_NAME AIIRPythonSupport
  ADD_TO_PARENT AIIRPythonSources.Core
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    Globals.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRTypes.cpp
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
################################################################################

add_aiir_python_common_capi_library(AIIRPythonCAPI
  INSTALL_COMPONENT AIIRPythonModules
  INSTALL_DESTINATION "${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_aiir_libs"
  OUTPUT_DIRECTORY "${AIIRPythonModules_ROOT_PREFIX}/_aiir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    AIIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    AIIRPythonSources
    AIIRPythonExtension.RegisterEverything
    ${_ADDL_TEST_SOURCES}
)

################################################################################
# Custom targets.
################################################################################

_flatten_aiir_python_targets(aiir_python_sources_deps AIIRPythonSources)

if(AIIR_PYTHON_STUBGEN_ENABLED)
  # _aiir stubgen
  # Note: All this needs to come before add_aiir_python_modules(AIIRPythonModules so that the install targets for the
  # generated type stubs get created.

  set(_core_type_stub_sources
    _aiir/__init__.pyi
    _aiir/ir.pyi
    _aiir/passmanager.pyi
    _aiir/rewrite.pyi
  )

  # Note 1: INTERFACE_SOURCES is a genex ($<BUILD_INTERFACE> $<INSTALL_INTERFACE>)
  # which will be evaluated by file(GENERATE ...) inside aiir_generate_type_stubs. This will evaluate to the correct
  # thing in the build dir (i.e., actual source dir paths) and in the install dir
  # (where it's a conventional path; see install/lib/cmake/aiir/AIIRTargets.cmake).
  #
  # Note 2: AIIRPythonExtension.Core is the target that is defined using target_sources(INTERFACE)
  # **NOT** AIIRPythonModules.extension._aiir.dso. So be sure to use the correct target!
  get_target_property(_core_extension_srcs AIIRPythonExtension.Core INTERFACE_SOURCES)

  # Why is MODULE_NAME _aiir here but aiir._aiir_libs._aiirPythonTestNanobind below???
  # The _aiir extension can be imported independently of any other python code and/or extension modules.
  # I.e., you could do `cd $AIIRPythonModules_ROOT_PREFIX/_aiir_libs && python -c "import _aiir"` (try it!).
  # _aiir is also (currently) the only extension for which this is possible because dialect extensions modules,
  # which generally make use of `aiir_value_subclass/aiir_type_subclass/aiir_attribute_subclass`, perform an
  # `import aiir` right when they're loaded (see the aiir_*_subclass ctors in NanobindAdaptors.h).
  # Note, this also why IMPORT_PATHS "${AIIRPythonModules_ROOT_PREFIX}/_aiir_libs" here while below
  # "${AIIRPythonModules_ROOT_PREFIX}/.." (because AIIR_BINDINGS_PYTHON_INSTALL_PREFIX, by default, ends at aiir).
  #
  # Further note: this function creates file targets like
  # "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_aiir_libs/_aiir/__init__.pyi". These must match the file targets
  # that declare_aiir_python_sources expects, which are like "${ROOT_DIR}/${WHATEVER_SOURCE}".
  # This is why _aiir_libs is prepended below.
  aiir_generate_type_stubs(
    MODULE_NAME _aiir
    DEPENDS_TARGETS AIIRPythonModules.extension._aiir.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_aiir_libs"
    OUTPUTS "${_core_type_stub_sources}"
    DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
    IMPORT_PATHS "${AIIRPythonModules_ROOT_PREFIX}/_aiir_libs"
  )
  set(_aiir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  list(TRANSFORM _core_type_stub_sources PREPEND "_aiir_libs/")
  # Note, we do not do ADD_TO_PARENT here so that the type stubs are not associated (as aiir_DEPENDS) with
  # AIIRPythonSources.Core (or something) when a distro is installed/created. Otherwise they would not be regenerated
  # by users of the distro (the stubs are still installed in the distro - they are just not added to aiir_DEPENDS).
  declare_aiir_python_sources(
    AIIRPythonExtension.Core.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    SOURCES "${_core_type_stub_sources}"
  )
  list(APPEND aiir_python_sources_deps AIIRPythonExtension.Core.type_stub_gen)

  # _aiirPythonTestNanobind stubgen

  if(AIIR_INCLUDE_TESTS)
    get_target_property(_test_extension_srcs AIIRPythonTestSources.PythonTestExtensionNanobind INTERFACE_SOURCES)
    aiir_generate_type_stubs(
      # This is the FQN path because dialect modules import _aiir when loaded. See above.
      MODULE_NAME aiir._aiir_libs._aiirPythonTestNanobind
      DEPENDS_TARGETS
        # You need both _aiir and _aiirPythonTestNanobind because dialect modules import _aiir when loaded
        # (so _aiir needs to be built before calling stubgen).
        AIIRPythonModules.extension._aiir.dso
        AIIRPythonModules.extension._aiirPythonTestNanobind.dso
        # You need this one so that ir.py "built" because aiir._aiir_libs.__init__.py import aiir.ir in _site_initialize.
        AIIRPythonModules.sources.AIIRPythonSources.Core.Python
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_aiir_libs"
      OUTPUTS _aiirPythonTestNanobind.pyi
      DEPENDS_TARGET_SRC_DEPS "${_test_extension_srcs}"
      IMPORT_PATHS "${AIIRPythonModules_ROOT_PREFIX}/.."
    )
    set(_aiirPythonTestNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")
    declare_aiir_python_sources(
      AIIRPythonTestSources.PythonTestExtensionNanobind.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT AIIRPythonTestSources.Dialects
      SOURCES _aiir_libs/_aiirPythonTestNanobind.pyi
    )
  endif()
endif()

add_custom_target("aiir-python-sources" DEPENDS ${aiir_python_sources_deps})
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-aiir-python-sources
    DEPENDS aiir-python-sources
    COMPONENT aiir-python-sources
  )
endif()

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

set(_declared_sources AIIRPythonSources AIIRPythonExtension.RegisterEverything)
if(AIIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _declared_sources AIIRPythonExtension.Core.type_stub_gen)
endif()

add_aiir_python_modules(AIIRPythonModules
  ROOT_PREFIX ${AIIRPythonModules_ROOT_PREFIX}
  INSTALL_PREFIX "${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    ${_declared_sources}
    ${_ADDL_TEST_SOURCES}
  COMMON_CAPI_LINK_LIBS
    AIIRPythonCAPI
)
if(AIIR_PYTHON_STUBGEN_ENABLED)
  add_dependencies(AIIRPythonModules "${_aiir_typestub_gen_target}")
  if(AIIR_INCLUDE_TESTS)
    add_dependencies(AIIRPythonModules "${_aiirPythonTestNanobind_typestub_gen_target}")
  endif()
endif()
