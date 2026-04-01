set(LLVM_TARGET_DEFINITIONS LoopExtensionOps.td)
aiir_tablegen(LoopExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(LoopExtensionOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectLoopExtensionOpsIncGen)

add_aiir_doc(LoopExtensionOps LoopExtensionOps Dialects/ -gen-op-doc -dialect=transform)
