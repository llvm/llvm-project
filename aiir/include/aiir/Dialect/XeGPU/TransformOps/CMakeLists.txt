set(LLVM_TARGET_DEFINITIONS XeGPUTransformOps.td)
aiir_tablegen(XeGPUTransformOps.h.inc -gen-op-decls)
aiir_tablegen(XeGPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIIRXeGPUTransformOpsIncGen)

add_aiir_doc(XeGPUTransformOps XeGPUTransformOps Dialects/ -gen-op-doc -dialect=transform)
