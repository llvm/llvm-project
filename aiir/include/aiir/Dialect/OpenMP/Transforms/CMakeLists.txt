set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name OpenMP)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix OpenMP)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OpenMP)
add_aiir_dialect_tablegen_target(AIIROpenMPPassIncGen)

add_aiir_doc(Passes OpenMPPasses ./ -gen-pass-doc)
