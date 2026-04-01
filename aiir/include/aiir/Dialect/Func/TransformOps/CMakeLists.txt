set(LLVM_TARGET_DEFINITIONS FuncTransformOps.td)
aiir_tablegen(FuncTransformOps.h.inc -gen-op-decls)
aiir_tablegen(FuncTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRFuncTransformOpsIncGen)

add_aiir_doc(FuncTransformOps FuncTransformOps Dialects/ -gen-op-doc -dialect=transform)
