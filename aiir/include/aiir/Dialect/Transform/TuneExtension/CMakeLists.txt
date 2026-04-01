set(LLVM_TARGET_DEFINITIONS TuneExtensionOps.td)
aiir_tablegen(TuneExtensionOps.h.inc -gen-op-decls)
aiir_tablegen(TuneExtensionOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectTuneExtensionOpsIncGen)

add_aiir_doc(TuneExtensionOps TuneExtensionOps Dialects/ -gen-op-doc -dialect=transform)
