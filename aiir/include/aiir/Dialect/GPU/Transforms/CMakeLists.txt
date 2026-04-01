set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name GPU)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPU)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPU)
add_aiir_dialect_tablegen_target(AIIRGPUPassIncGen)

add_aiir_doc(Passes GPUPasses ./ -gen-pass-doc)
