set(LLVM_TARGET_DEFINITIONS AffineTransformOps.td)
aiir_tablegen(AffineTransformOps.h.inc -gen-op-decls)
aiir_tablegen(AffineTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRAffineTransformOpsIncGen)

add_aiir_doc(AffineTransformOps AffineLoopTransformOps Dialects/ -gen-op-doc -dialect=transform)
