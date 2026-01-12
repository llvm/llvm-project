# This replicates part of the add_mlir_dialect cmake function from MLIR that
# cannot be used here. This happens because it expects to be run inside MLIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds).

# Equivalent to add_mlir_dialect(CIROps cir)
set(LLVM_TARGET_DEFINITIONS CIROps.td)
mlir_tablegen(CIROps.h.inc -gen-op-decls)
mlir_tablegen(CIROps.cpp.inc -gen-op-defs)
mlir_tablegen(CIROpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CIROpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(CIROpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CIROpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRCIROpsIncGen)
add_dependencies(mlir-headers MLIRCIROpsIncGen)

mlir_tablegen(CIROpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CIROpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CIROpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CIROpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRCIREnumsGen)

clang_tablegen(CIRLowering.inc -gen-cir-lowering
               SOURCE CIROps.td
               TARGET CIRLowering)

set(LLVM_TARGET_DEFINITIONS CIRTypeConstraints.td)
mlir_tablegen(CIRTypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(CIRTypeConstraints.cpp.inc -gen-type-constraint-defs)
add_public_tablegen_target(MLIRCIRTypeConstraintsIncGen)
add_dependencies(mlir-headers MLIRCIRTypeConstraintsIncGen)
