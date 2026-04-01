set(LLVM_TARGET_DEFINITIONS PDLExtensionOps.td)
aiir_tablegen(PDLExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(PDLExtensionOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectPDLExtensionOpsIncGen)

add_aiir_doc(PDLExtensionOps PDLExtensionOps Dialects/ -gen-op-doc -dialect=transform)
