set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name SCF)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix SCF)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix SCF)
add_aiir_dialect_tablegen_target(AIIRSCFPassIncGen)

add_aiir_doc(Passes SCFPasses ./ -gen-pass-doc)

