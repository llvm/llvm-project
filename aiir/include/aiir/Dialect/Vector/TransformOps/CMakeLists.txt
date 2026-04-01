set(LLVM_TARGET_DEFINITIONS VectorTransformOps.td)
aiir_tablegen(VectorTransformOps.h.inc -gen-op-decls)
aiir_tablegen(VectorTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRVectorTransformOpsIncGen)

add_aiir_doc(VectorTransformOps VectorTransformOps Dialects/ -gen-op-doc)
