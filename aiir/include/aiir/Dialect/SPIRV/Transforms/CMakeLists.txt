set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name SPIRV)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix SPIRV)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix SPIRV)
add_aiir_dialect_tablegen_target(AIIRSPIRVPassIncGen)

add_aiir_doc(Passes SPIRVPasses ./ -gen-pass-doc)
