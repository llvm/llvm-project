set(LLVM_TARGET_DEFINITIONS XeMachine.td)
mlir_tablegen(XeMachineDialect.h.inc -gen-dialect-decls)
mlir_tablegen(XeMachineDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(XeMachineEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeMachineEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(XeMachineAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(XeMachineAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(XeMachineTypes.h.inc -gen-typedef-decls)
mlir_tablegen(XeMachineTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(XeMachineOps.h.inc -gen-op-decls)
mlir_tablegen(XeMachineOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS XeMachineTransformOps.td)
mlir_tablegen(XeMachineTransformOps.h.inc -gen-op-decls)
mlir_tablegen(XeMachineTransformOps.cpp.inc -gen-op-defs)

# The op-interface generator ignores records outside the top-level input
# file, so the interfaces .td must be tablegen'd on its own.
set(LLVM_TARGET_DEFINITIONS XeMachineInterfaces.td)
mlir_tablegen(XeMachineInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(XeMachineInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(InterXeMachineIRIncGen)
