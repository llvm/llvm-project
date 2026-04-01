add_aiir_dialect(AMDGPU amdgpu)
add_aiir_doc(AMDGPU AMDGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AMDGPUEnums.td)
aiir_tablegen(AMDGPUEnums.h.inc -gen-enum-decls)
aiir_tablegen(AMDGPUEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRAMDGPUEnumsGen)

set(LLVM_TARGET_DEFINITIONS AMDGPUAttrs.td)
aiir_tablegen(AMDGPUAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=amdgpu)
aiir_tablegen(AMDGPUAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=amdgpu)
add_aiir_dialect_tablegen_target(AIIRAMDGPUAttributesIncGen)
