set(LLVM_TARGET_DEFINITIONS WasmSSATypes.td)
aiir_tablegen(WasmSSATypeConstraints.h.inc -gen-type-constraint-decls)
aiir_tablegen(WasmSSATypeConstraints.cpp.inc -gen-type-constraint-defs)

set (LLVM_TARGET_DEFINITIONS WasmSSAInterfaces.td)
aiir_tablegen(WasmSSAInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(WasmSSAInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRWasmSSAInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS WasmSSAOps.td)

add_aiir_dialect(WasmSSAOps wasmssa)
add_aiir_doc(WasmSSAOps WasmSSAOps Dialects/ -gen-dialect-doc -dialect wasmssa)
