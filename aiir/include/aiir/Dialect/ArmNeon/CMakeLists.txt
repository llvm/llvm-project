add_aiir_dialect(ArmNeon arm_neon)
add_aiir_doc(ArmNeon ArmNeon Dialects/ -gen-dialect-doc -dialect=arm_neon)

set(LLVM_TARGET_DEFINITIONS ArmNeon.td)
aiir_tablegen(ArmNeonConversions.inc -gen-llvmir-conversions)
add_aiir_dialect_tablegen_target(AIIRArmNeonConversionsIncGen)

add_subdirectory(TransformOps)
