set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name XeGPU)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix XeGPU)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix XeGPU)
add_aiir_dialect_tablegen_target(AIIRXeGPUPassIncGen)

add_aiir_doc(Passes XeGPUPasses ./ -gen-pass-doc)
