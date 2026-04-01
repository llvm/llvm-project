set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Func)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Func)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Func)
add_aiir_dialect_tablegen_target(AIIRFuncTransformsIncGen)

add_aiir_doc(Passes FuncPasses ./ -gen-pass-doc)
