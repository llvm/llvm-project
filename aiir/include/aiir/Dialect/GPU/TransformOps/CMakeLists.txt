set(LLVM_TARGET_DEFINITIONS GPUTransformOps.td)
aiir_tablegen(GPUTransformOps.h.inc -gen-op-decls)
aiir_tablegen(GPUTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRGPUTransformOpsIncGen)

add_aiir_doc(GPUTransformOps GPUTransformOps Dialects/ -gen-op-doc -dialect=transform)
