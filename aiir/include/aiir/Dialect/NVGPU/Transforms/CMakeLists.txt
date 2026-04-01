set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name NVGPU)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix NVGPU)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix NVGPU)
add_aiir_dialect_tablegen_target(AIIRNVGPUPassIncGen)

add_aiir_doc(Passes NVGPUPasses ./ -gen-pass-doc)
