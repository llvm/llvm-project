set(LLVM_TARGET_DEFINITIONS SMTExtensionOps.td)
aiir_tablegen(SMTExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(SMTExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIIRTransformDialectSMTExtensionOpsIncGen)

add_aiir_doc(SMTExtensionOps SMTExtensionOps Dialects/ -gen-op-doc -dialect=transform)
