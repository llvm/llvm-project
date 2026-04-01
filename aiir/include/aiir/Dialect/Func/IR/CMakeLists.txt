set(LLVM_TARGET_DEFINITIONS FuncOps.td)
aiir_tablegen(FuncOps.h.inc -gen-op-decls)
aiir_tablegen(FuncOps.cpp.inc -gen-op-defs)
aiir_tablegen(FuncOpsDialect.h.inc -gen-dialect-decls)
aiir_tablegen(FuncOpsDialect.cpp.inc -gen-dialect-defs)
add_aiir_dialect_tablegen_target(AIIRFuncOpsIncGen)

add_aiir_doc(FuncOps FuncOps Dialects/ -gen-op-doc -dialect=func)
