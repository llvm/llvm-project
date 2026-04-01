set(LLVM_TARGET_DEFINITIONS CGOps.td)
aiir_tablegen(CGOps.h.inc -gen-op-decls)
aiir_tablegen(CGOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CGOpsIncGen)
