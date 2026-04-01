add_aiir_dialect(NVGPU nvgpu)
add_aiir_doc(NVGPUOps NVGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS NVGPUOps.td)
aiir_tablegen(NVGPUOps.h.inc -gen-op-decls)
aiir_tablegen(NVGPUOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRNVGPUOpsIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
aiir_tablegen(NVGPUEnums.h.inc -gen-enum-decls)
aiir_tablegen(NVGPUEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRNVGPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
aiir_tablegen(NVGPUAttrDefs.h.inc -gen-attrdef-decls)
aiir_tablegen(NVGPUAttrDefs.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRNVGPUAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPUTypes.td)
aiir_tablegen(NVGPUTypeDefs.h.inc -gen-typedef-decls)
aiir_tablegen(NVGPUTypeDefs.cpp.inc -gen-typedef-defs)
add_aiir_dialect_tablegen_target(AIIRNVGPUTypesIncGen)
