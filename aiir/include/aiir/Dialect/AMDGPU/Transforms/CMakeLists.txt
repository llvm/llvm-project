set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name AMDGPU)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix AMDGPU)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix AMDGPU)
add_aiir_dialect_tablegen_target(AIIRAMDGPUTransformsIncGen)

add_aiir_doc(Passes AMDGPUPasses ./ -gen-pass-doc)
