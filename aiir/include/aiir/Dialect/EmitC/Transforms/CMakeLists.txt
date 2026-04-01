set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix EmitC)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix EmitC)
add_aiir_dialect_tablegen_target(AIIREmitCTransformsIncGen)

add_aiir_doc(Passes EmitCPasses ./ -gen-pass-doc)
