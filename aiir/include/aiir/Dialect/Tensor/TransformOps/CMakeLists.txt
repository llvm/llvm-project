set(LLVM_TARGET_DEFINITIONS TensorTransformOps.td)
aiir_tablegen(TensorTransformOps.h.inc -gen-op-decls)
aiir_tablegen(TensorTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRTensorTransformOpsIncGen)

add_aiir_doc(TensorTransformOps TensorTransformOps Dialects/ -gen-op-doc -dialect=transform)
