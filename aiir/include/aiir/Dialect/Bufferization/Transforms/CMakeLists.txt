set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Bufferization)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Bufferization)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Bufferization)
add_aiir_dialect_tablegen_target(AIIRBufferizationPassIncGen)

add_aiir_doc(Passes BufferizationPasses ./ -gen-pass-doc)
