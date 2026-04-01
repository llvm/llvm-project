set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Shape)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Shape)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Shape)
add_aiir_dialect_tablegen_target(AIIRShapeTransformsIncGen)

add_aiir_doc(Passes ShapePasses ./ -gen-pass-doc)
