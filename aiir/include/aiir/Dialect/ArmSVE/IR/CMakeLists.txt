add_aiir_dialect(ArmSVE arm_sve ArmSVE)
add_aiir_doc(ArmSVE ArmSVE Dialects/ -gen-dialect-doc -dialect=arm_sve)

set(LLVM_TARGET_DEFINITIONS ArmSVE.td)
aiir_tablegen(ArmSVEConversions.inc -gen-llvmir-conversions)
add_aiir_dialect_tablegen_target(AIIRArmSVEConversionsIncGen)
