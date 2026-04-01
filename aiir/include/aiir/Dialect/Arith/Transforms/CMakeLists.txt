set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Arith)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Arith)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Arith)
add_aiir_dialect_tablegen_target(AIIRArithTransformsIncGen)

add_aiir_doc(Passes ArithPasses ./ -gen-pass-doc)
