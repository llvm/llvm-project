add_aiir_dialect(EmitC emitc)
add_aiir_doc(EmitC EmitC Dialects/ -gen-dialect-doc -dialect emitc)

set(LLVM_TARGET_DEFINITIONS EmitCInterfaces.td)
aiir_tablegen(EmitCInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(EmitCInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_generic_tablegen_target(AIIREmitCInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS EmitCAttributes.td)
aiir_tablegen(EmitCEnums.h.inc -gen-enum-decls)
aiir_tablegen(EmitCEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(EmitCAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(EmitCAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIREmitCAttributesIncGen)
