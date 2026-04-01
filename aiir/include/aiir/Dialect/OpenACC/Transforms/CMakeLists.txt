set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name OpenACC)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix OpenACC)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OpenACC)
add_aiir_dialect_tablegen_target(AIIROpenACCPassIncGen)

add_aiir_doc(Passes OpenACCPasses ./ -gen-pass-doc)
