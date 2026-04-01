set(LLVM_TARGET_DEFINITIONS ArmNeonVectorTransformOps.td)
aiir_tablegen(ArmNeonVectorTransformOps.h.inc -gen-op-decls)
aiir_tablegen(ArmNeonVectorTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRArmNeonVectorTransformOpsIncGen)

add_aiir_doc(ArmNeonVectorTransformOps ArmNeonVectorTransformOps Dialects/ -gen-op-doc -dialect=transform)
