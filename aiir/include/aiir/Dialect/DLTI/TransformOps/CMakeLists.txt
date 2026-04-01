set(LLVM_TARGET_DEFINITIONS DLTITransformOps.td)
aiir_tablegen(DLTITransformOps.h.inc -gen-op-decls)
aiir_tablegen(DLTITransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRDLTITransformOpsIncGen)

add_aiir_doc(DLTITransformOps DLTITransformOps Dialects/ -gen-op-doc -dialect=transform)
