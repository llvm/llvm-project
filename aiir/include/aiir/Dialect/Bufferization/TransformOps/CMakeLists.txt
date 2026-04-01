set(LLVM_TARGET_DEFINITIONS BufferizationTransformOps.td)
aiir_tablegen(BufferizationTransformOps.h.inc -gen-op-decls)
aiir_tablegen(BufferizationTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRBufferizationTransformOpsIncGen)

add_aiir_doc(BufferizationTransformOps BufferizationTransformOps Dialects/ -gen-op-doc -dialect=transform)
