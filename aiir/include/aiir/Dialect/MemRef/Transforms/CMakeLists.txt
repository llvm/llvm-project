set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name MemRef)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix MemRef)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix MemRef)
add_aiir_dialect_tablegen_target(AIIRMemRefPassIncGen)

add_aiir_doc(Passes MemRefPasses ./ -gen-pass-doc)
