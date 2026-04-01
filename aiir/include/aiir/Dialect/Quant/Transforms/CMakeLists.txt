set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Quant)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Quant)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Quant)
add_aiir_dialect_tablegen_target(AIIRQuantTransformsIncGen)

add_aiir_doc(Passes QuantPasses ./ -gen-pass-doc)
