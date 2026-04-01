# This replicates part of the add_aiir_dialect cmake function from AIIR that
# cannot be used here. This happens because it expects to be run inside AIIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds).

# Equivalent to add_aiir_dialect(CIROps cir)
set(LLVM_TARGET_DEFINITIONS CIROps.td)
aiir_tablegen(CIROps.h.inc -gen-op-decls)
aiir_tablegen(CIROps.cpp.inc -gen-op-defs)
aiir_tablegen(CIROpsTypes.h.inc -gen-typedef-decls)
aiir_tablegen(CIROpsTypes.cpp.inc -gen-typedef-defs)
aiir_tablegen(CIROpsDialect.h.inc -gen-dialect-decls)
aiir_tablegen(CIROpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AIIRCIROpsIncGen)
add_dependencies(aiir-headers AIIRCIROpsIncGen)

aiir_tablegen(CIROpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(CIROpsEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(CIROpsAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(CIROpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AIIRCIREnumsGen)

clang_tablegen(CIRLowering.inc -gen-cir-lowering
               SOURCE CIROps.td
               TARGET CIRLowering)

set(LLVM_TARGET_DEFINITIONS CIRTypeConstraints.td)
aiir_tablegen(CIRTypeConstraints.h.inc -gen-type-constraint-decls)
aiir_tablegen(CIRTypeConstraints.cpp.inc -gen-type-constraint-defs)
add_public_tablegen_target(AIIRCIRTypeConstraintsIncGen)
add_dependencies(aiir-headers AIIRCIRTypeConstraintsIncGen)
