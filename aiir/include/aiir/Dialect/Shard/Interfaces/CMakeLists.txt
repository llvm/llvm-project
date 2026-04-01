set(LLVM_TARGET_DEFINITIONS ShardingInterface.td)
aiir_tablegen(ShardingInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(ShardingInterface.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRShardingInterfaceIncGen)
