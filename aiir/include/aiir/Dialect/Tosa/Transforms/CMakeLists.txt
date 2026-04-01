set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Tosa)
aiir_tablegen(PassesEnums.h.inc -gen-enum-decls)
aiir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Tosa)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Tosa)
add_aiir_dialect_tablegen_target(AIIRTosaPassIncGen)

add_aiir_doc(Passes TosaPasses ./ -gen-pass-doc)
