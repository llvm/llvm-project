add_aiir_doc(ShardOps ShardOps Dialects/ -gen-op-doc -dialect=shard)
add_aiir_doc(ShardOps ShardAttrs Dialects/ -gen-attrdef-doc -dialect=shard)

set(LLVM_TARGET_DEFINITIONS ShardOps.td)
aiir_tablegen(ShardDialect.cpp.inc -gen-dialect-defs -dialect=shard)
aiir_tablegen(ShardDialect.h.inc -gen-dialect-decls -dialect=shard)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
aiir_tablegen(ShardAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(ShardAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
aiir_tablegen(ShardEnums.h.inc -gen-enum-decls)
aiir_tablegen(ShardEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
aiir_tablegen(ShardTypes.h.inc -gen-typedef-decls)
aiir_tablegen(ShardTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ShardOps.td)
aiir_tablegen(ShardOps.h.inc -gen-op-decls)
aiir_tablegen(ShardOps.cpp.inc -gen-op-defs)

add_aiir_dialect_tablegen_target(AIIRShardIncGen)
