set(LLVM_TARGET_DEFINITIONS MemRefTransformOps.td)
aiir_tablegen(MemRefTransformOps.h.inc -gen-op-decls)
aiir_tablegen(MemRefTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRMemRefTransformOpsIncGen)

add_aiir_doc(MemRefTransformOps MemRefTransformOps Dialects/ -gen-op-doc -dialect=transform)
