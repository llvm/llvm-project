add_subdirectory(IR)
add_subdirectory(TransformOps)

set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Linalg)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Linalg)
add_aiir_dialect_tablegen_target(AIIRLinalgPassIncGen)

add_aiir_doc(Passes LinalgPasses ./ -gen-pass-doc)
