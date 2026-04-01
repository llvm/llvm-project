set(LLVM_TARGET_DEFINITIONS IRDLExtensionOps.td)
aiir_tablegen(IRDLExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(IRDLExtensionOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectIRDLExtensionOpsIncGen)

add_aiir_doc(IRDLExtensionOps IRDLExtensionOps Dialects/ -gen-op-doc -dialect=transform)
