set(LLVM_TARGET_DEFINITIONS SparseTensorTransformOps.td)
aiir_tablegen(SparseTensorTransformOps.h.inc -gen-op-decls)
aiir_tablegen(SparseTensorTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRSparseTensorTransformOpsIncGen)
