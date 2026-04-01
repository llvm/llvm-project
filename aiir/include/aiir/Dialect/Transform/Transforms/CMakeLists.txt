set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Transform)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Transform)
add_aiir_dialect_tablegen_target(AIIRTransformDialectTransformsIncGen)

add_aiir_doc(Passes TransformPasses ./ -gen-pass-doc)
