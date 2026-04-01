set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Shard)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Shard)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Shard)
add_aiir_dialect_tablegen_target(AIIRShardPassIncGen)

add_aiir_doc(Passes ShardPasses ./ -gen-pass-doc)
