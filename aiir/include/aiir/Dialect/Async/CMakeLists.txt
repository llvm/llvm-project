add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Async)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Async)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Async)
add_aiir_dialect_tablegen_target(AIIRAsyncPassIncGen)

add_aiir_doc(Passes AsyncPasses ./ -gen-pass-doc)
