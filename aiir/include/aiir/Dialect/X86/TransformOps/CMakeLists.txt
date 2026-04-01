set(LLVM_TARGET_DEFINITIONS X86TransformOps.td)
aiir_tablegen(X86TransformOps.h.inc -gen-op-decls)
aiir_tablegen(X86TransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRX86TransformOpsIncGen)
