set(LLVM_TARGET_DEFINITIONS SCFTransformOps.td)
aiir_tablegen(SCFTransformOps.h.inc -gen-op-decls)
aiir_tablegen(SCFTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRSCFTransformOpsIncGen)

add_aiir_doc(SCFTransformOps SCFLoopTransformOps Dialects/ -gen-op-doc -dialect=transform)
