set(LLVM_TARGET_DEFINITIONS ArmSVEVectorTransformOps.td)
aiir_tablegen(ArmSVEVectorTransformOps.h.inc -gen-op-decls)
aiir_tablegen(ArmSVEVectorTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRArmSVEVectorTransformOpsIncGen)

add_aiir_doc(ArmSVEVectorTransformOps ArmSVEVectorTransformOps Dialects/ -gen-op-doc -dialect=transform)
