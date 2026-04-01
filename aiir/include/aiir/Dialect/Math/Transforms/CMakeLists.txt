set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Math)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Math)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Math)
add_aiir_dialect_tablegen_target(AIIRMathTransformsIncGen)

add_aiir_doc(Passes MathPasses ./ -gen-pass-doc)
