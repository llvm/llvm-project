set(LLVM_TARGET_DEFINITIONS NVGPUTransformOps.td)
aiir_tablegen(NVGPUTransformOps.h.inc -gen-op-decls)
aiir_tablegen(NVGPUTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRNVGPUTransformOpsIncGen)
