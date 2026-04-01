set(LLVM_TARGET_DEFINITIONS DebugExtensionOps.td)
aiir_tablegen(DebugExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(DebugExtensionOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectDebugExtensionOpsIncGen)

add_aiir_doc(DebugExtensionOps DebugExtensionOps Dialects/ -gen-op-doc -dialect=transform)
