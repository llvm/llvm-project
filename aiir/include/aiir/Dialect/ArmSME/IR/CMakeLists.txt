add_aiir_dialect(ArmSME arm_sme ArmSME)

# Generate declarations and definitions of ArmSME Ops
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
aiir_tablegen(ArmSMEOps.h.inc -gen-op-decls)
aiir_tablegen(ArmSMEOps.cpp.inc -gen-op-defs)
aiir_tablegen(ArmSMEEnums.h.inc -gen-enum-decls)
aiir_tablegen(ArmSMEEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(ArmSMEAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=arm_sme)
aiir_tablegen(ArmSMEAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=arm_sme)
add_aiir_dialect_tablegen_target(AIIRArmSMEOpsIncGen)

# Generate LLVM IR Conversions
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
aiir_tablegen(ArmSMEOpsConversions.inc -gen-llvmir-conversions)
add_aiir_dialect_tablegen_target(AIIRArmSMEConversionsIncGen)

# Generate op interface declarations and definitions
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
aiir_tablegen(ArmSMEOpInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(ArmSMEOpInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRArmSMEOpInterfaces)

# Generate declarations and definitions of ArmSME intrinsic Ops
set(LLVM_TARGET_DEFINITIONS ArmSMEIntrinsicOps.td)
aiir_tablegen(ArmSMEIntrinsicOps.h.inc -gen-op-decls)
aiir_tablegen(ArmSMEIntrinsicOps.cpp.inc -gen-op-defs)
aiir_tablegen(ArmSMEIntrinsicConversions.inc -gen-llvmir-conversions)
add_aiir_dialect_tablegen_target(AIIRArmSMEIntrinsicOpsIncGen)

# Generate the docs
add_aiir_doc(ArmSMEOps ArmSMEOps Dialects/ -gen-op-doc -dialect=arm_sme)
add_aiir_doc(ArmSMEIntrinsicOps ArmSMEIntrinsicOps Dialects/ -gen-op-doc -dialect=arm_sme)
