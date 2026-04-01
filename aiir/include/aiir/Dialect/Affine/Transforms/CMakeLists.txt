set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Affine)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Affine)
add_aiir_dialect_tablegen_target(AIIRAffinePassIncGen)

add_aiir_doc(Passes AffinePasses ./ -gen-pass-doc)
