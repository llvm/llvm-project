set(LLVM_TARGET_DEFINITIONS CGOps.td)
mlir_tablegen(CGOps.h.inc -gen-op-decls)
mlir_tablegen(CGOps.cpp.inc -gen-op-defs)
add_flang_tablegen_target(CGOpsIncGen)
